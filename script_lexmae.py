from peach.datasets.dataset_long_text import DatasetLongText, add_data_hyperparameters
import os.path
import tqdm
from peach.enc_utils.enc_learners import LearnerMixin
from peach.nn_utils.masked_lm import mlm_input_ids_masking_onthefly, text_part_mask_generation
from peach.base import *
from peach.models.modeling_lexmae import add_model_hyperparameters, \
    BertForEDMLM, DistilBertForEDMLM, RobertaForEDMLM


def add_training_hyperparameters(parser):
    parser.add_argument("--enc_mlm_prob", type=float, default=0.00)
    parser.add_argument("--dec_mlm_prob", type=float, default=0.50)
    parser.add_argument("--dec_mlm_overlap", type=str, default="inclusive",
                        choices=["random", "inclusive", "exclusive"])  # dec masks include enc ones

    parser.add_argument("--mlm_enc_loss_weight", type=float, default=1.0)
    parser.add_argument("--mlm_dec_loss_weight", type=float, default=1.0)


class LexmaeLearner(LearnerMixin):
    def __init__(self, model_args, config, tokenizer, encoder, query_encoder=None):
        super(LexmaeLearner, self).__init__(model_args, config, tokenizer, encoder, query_encoder, )
        self.mask_token_id = self.tokenizer.mask_token_id
        self.special_token_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id, ]
        self.vocab_size = len(self.tokenizer)
        self.base_model_prefix = "encoder." + encoder.base_model_prefix

    def forward(
            self,
            input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            masked_input_ids=None, masked_flags=None, mlm_labels=None,
            dec_masked_input_ids=None, dec_attention_mask=None, dec_masked_flags=None, dec_mlm_labels=None,
            training_progress=None, training_mode=None, **kwargs,
    ):
        if training_mode is None:
            raise self.encoder(input_ids, attention_mask, token_type_ids, position_ids,)

        # embedding
        dict_for_meta, dict_for_loss = {}, {}

        dict_for_meta["mlm_enc_loss_weight"] = self.model_args.mlm_enc_loss_weight
        dict_for_meta["mlm_dec_loss_weight"] = self.model_args.mlm_dec_loss_weight

        # encoder masking and forward
        if masked_input_ids is None:  # on-the-fly generation
            masked_input_ids, masked_flags, mlm_labels, rdm_probs = mlm_input_ids_masking_onthefly(
                input_ids, attention_mask, self.mask_token_id, self.special_token_ids,
                mlm_prob=self.model_args.enc_mlm_prob, mlm_rdm_prob=0.1, mlm_keep_prob=0.1,
                vocab_size=self.vocab_size, external_rdm_probs=None, extra_masked_flags=None,
                resample_nonmask=False,)
            enc_mlm_prob = self.model_args.enc_mlm_prob
        else:
            enc_mlm_prob = self.model_args.data_mlm_prob

        enc_outputs = self.encoder(
            masked_input_ids, attention_mask, token_type_ids, position_ids,
            enc_mlm_labels=mlm_labels, disable_encoding=False, disable_decoding=True, )
        dict_for_loss["mlm_enc_loss"] = enc_outputs["loss"]

        # decoder masking and forward
        if dec_masked_input_ids is None:
            if self.model_args.dec_mlm_overlap == "random":
                extra_masked_flags, exclude_flags = None, None
                dec_mlm_prob = self.model_args.dec_mlm_prob
            elif self.model_args.dec_mlm_overlap == "inclusive":
                assert self.model_args.dec_mlm_prob >= enc_mlm_prob
                extra_masked_flags, exclude_flags = masked_flags, None
                dec_mlm_prob = (self.model_args.dec_mlm_prob - enc_mlm_prob) / (1. - enc_mlm_prob)
            elif self.model_args.dec_mlm_overlap == "exclusive":
                assert self.model_args.dec_mlm_prob <= (1.0 - enc_mlm_prob)
                extra_masked_flags, exclude_flags = None, masked_flags
                dec_mlm_prob = self.model_args.dec_mlm_prob / (1.0 - enc_mlm_prob)
            else:
                raise NotImplementedError

            dec_masked_input_ids, dec_masked_flags, dec_mlm_labels, _ = mlm_input_ids_masking_onthefly(
                input_ids, attention_mask, self.mask_token_id, self.special_token_ids,
                mlm_prob=dec_mlm_prob, mlm_rdm_prob=0.1, mlm_keep_prob=0.1, vocab_size=self.vocab_size,
                external_rdm_probs=None, extra_masked_flags=extra_masked_flags, exclude_flags=exclude_flags,
                resample_nonmask=True, )

        dec_attention_mask = dec_attention_mask if dec_attention_mask is not None else attention_mask
        dec_outputs = self.encoder(
            dec_input_ids=dec_masked_input_ids, dec_attention_mask=dec_attention_mask,
            dec_token_type_ids=None, dec_position_ids=None,
            enc_cls_rep=enc_outputs["sentence_embedding"], enc_hidden_states=enc_outputs["hidden_states"],
            dec_mlm_labels=dec_mlm_labels, disable_encoding=True, disable_decoding=False, )
        dict_for_loss["mlm_dec_loss"] = dec_outputs["dec_loss"]

        loss = 0.
        for k in dict_for_loss:
            if k + "_weight" in dict_for_meta:
                if dict_for_meta[k + "_weight"] == 0.:
                    loss += 0.0  # save calc
                else:
                    loss += dict_for_meta[k + "_weight"] * dict_for_loss[k]
            else:
                loss += dict_for_loss[k]
        dict_for_loss["loss"] = loss

        dict_for_meta.update(dict_for_loss)
        return dict_for_meta


def train(args, train_dataset, model, accelerator, tokenizer, eval_dataset=None, eval_fn=None):
    if accelerator.is_local_main_process:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    else:
        tb_writer = None

    train_dataloader = setup_train_dataloader(args, train_dataset, accelerator)
    model, optimizer, lr_scheduler = setup_opt(args, model, accelerator, len(train_dataloader))

    logging_berfore_training(args, train_dataset)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    step_loss = 0.
    step_loss_dict = defaultdict(float)
    best_metric = NEG_INF
    ma_dict = MovingAverageDict()
    model.train()
    model.zero_grad()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # print(dist.get_rank(), train_dataloader["input_ids"].shape[0])
            step += 1  # fix for accumulation
            # sync_context = model.no_sync if accelerator.distributed_type != accelerate.DistributedType.NO and \
            #                                 step % args.gradient_accumulation_steps > 0 else nullcontext
            #
            # with sync_context():  # disable DDP sync for accumulation step
            outputs = model(training_mode="pre-training", **batch)
            update_wrt_loss(args, accelerator, model, optimizer, outputs["loss"])
            # update
            for key in outputs:
                if key.endswith("loss"):
                    step_loss_dict[key] += outputs[key].item() / args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader):

                model_update_wrt_gradient(args, accelerator, model, optimizer, lr_scheduler)

                # update loss for logging
                if accelerator.is_local_main_process:
                    # tensorboard
                    if tb_writer is not None and (  # local main process
                            args.tensorboard_steps > 0 and global_step % args.tensorboard_steps == 0):
                        for key, loss_val in step_loss_dict.items():
                            tb_writer.add_scalar(f"training-{key}", loss_val, global_step)
                        for key, elem in outputs.items():
                            if (key not in step_loss_dict) and isinstance(elem, (int, float)):
                                tb_writer.add_scalar(f"training-meta-{key}", elem, global_step)
                    # log fire
                    ma_dict(step_loss_dict)
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logging.info(f"Log at step-{global_step}: {ma_dict.get_val_str()}")
                step_loss_dict = defaultdict(float)

                # assert args.save_steps > 0, "save_steps should be larger than 0 when no dev"
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_with_default_name(
                        args, accelerator, args.output_dir, model, tokenizer, args, save_specified_module="encoder", )

                # DDP for eval
                if eval_fn is not None and args.eval_steps > 0 and (
                        global_step % args.eval_steps == 0 or global_step == args.max_train_steps - 1):
                    key_metric, eval_metrics = eval_fn(
                        args, eval_dataset, model, accelerator, global_step=global_step,
                        tb_writer=tb_writer, tokenizer=tokenizer, key_metric_name=args.dev_key_metric,
                        similarity_metric=None, query_encoder=None,
                        use_accelerator=True,
                    )
                    if key_metric >= best_metric:  # always false in sub process
                        best_metric = key_metric
                        save_model_with_default_name(
                            args, accelerator, args.output_dir, model, tokenizer, args_to_save=args,
                            wait_for_everyone=False, save_specified_module="encoder",
                            # do not wait other subprocesses during saving in main process
                        )
                accelerator.wait_for_everyone()  # other subprocesses must wait the main process

                progress_bar.update(1)
                global_step += 1  # update global step after determine whether eval

                if global_step >= args.max_train_steps:
                    break
        if not (eval_fn is not None and args.eval_steps > 0):
            save_model_with_default_name(
                args, accelerator, args.output_dir, model, tokenizer, args,
                save_specified_module="encoder", )
        else:
            save_model_with_default_name(
                args, accelerator, os.path.join(args.output_dir, "last_checkpoint"), model, tokenizer, args,
                save_specified_module="encoder", )
    if eval_fn is not None and args.eval_steps > 0:
        save_model_with_default_name(
            args, accelerator, os.path.join(args.output_dir, "last_checkpoint"), model, tokenizer, args,
            save_specified_module="encoder", )
    model.zero_grad()
    model.eval()
    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser()
    # add task specific hyparam
    #
    define_hparams_training(parser)

    parser.add_argument("--data_load_type", type=str, default="disk", choices=["disk", "memory"])
    parser.add_argument("--data_dir", type=str, default=USER_HOME + "/ws/data/set/")
    parser.add_argument("--dev_key_metric", type=str, default="none")

    parser.add_argument("--encoder", type=str, default=None)

    model_param_list = add_model_hyperparameters(parser)
    add_data_hyperparameters(parser)
    add_training_hyperparameters(parser)

    args = parser.parse_args()
    accelerator = setup_prerequisite(args)

    config, tokenizer = load_config_and_tokenizer(
        args, config_kwargs={
            # "problem_type": args.problem_type,
            # "num_labels": num_labels,
        })
    for param in model_param_list:
        setattr(config, param, getattr(args, param))

    encoder_class = BertForEDMLM
    if args.encoder is not None:
        if args.encoder == "roberta":
            encoder_class = RobertaForEDMLM
        elif args.encoder == "distilbert":
            encoder_class = DistilBertForEDMLM
        elif args.encoder == "bert":
            encoder_class = BertForEDMLM
        else:
            raise AttributeError(args.encoder)
    # if args.encoder_type is not None and args.encoder_type != None:
    #     if args.encoder_type == "condenser":
    #         encoder_class = BertForCondenser
    #     else:
    #         raise NotImplementedError(args.encoder_type)
    encoder = encoder_class.from_pretrained(args.model_name_or_path, config=config)

    if args.do_train:
        model = LexmaeLearner(args, config, tokenizer, encoder)
        with accelerator.main_process_first():
            train_dataset = DatasetLongText(
                "train", args.data_dir, args.data_rel_paths, args.data_load_type, args, tokenizer)
        with accelerator.main_process_first():
            # dev_dataset = DatasetCustomRerank(
            #     args.dev_type, args.data_dir, "memory", args, tokenizer, num_dev=args.num_dev,
            #     add_title=(not args.no_title),
            #     filepath_dev_qid2top1000pids=args.eval_reranking_source,
            # )
            dev_dataset = None

        train(
            args, train_dataset, model, accelerator, tokenizer,
            eval_dataset=dev_dataset, eval_fn=None)


if __name__ == '__main__':
    main()









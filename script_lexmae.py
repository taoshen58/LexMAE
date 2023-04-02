import json
import tqdm

from peach.base import *

from peach.datasets.marco.dataset_marco_passages import DatasetMarcoPassagesRanking
from peach.datasets.marco.dataset_marco_eval import DatasetRerank, DatasetCustomRerank

from peach.enc_utils.eval_functions import evaluate_encoder_reranking
from peach.enc_utils.eval_sparse import evaluate_sparse_retreival
from peach.enc_utils.hn_gen_sparse import get_hard_negative_by_sparse_retrieval

from peach.models.modeling_splade_series import add_model_hyperparameters, DistilBertSpladeEnocder, \
    BertSpladeEnocder, RobertaSpladeEnocder
from peach.enc_utils.enc_learners import LearnerMixin, FLOPS

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs
from peach.common import load_pickle

from peach.nn_utils.general import combine_dual_inputs_by_attention_mask

def add_training_hyperparameters(parser):
    parser.add_argument("--lambda_d", type=float, default=0.0008)
    parser.add_argument("--lambda_q", type=float, default=None)
    parser.add_argument("--lambda_ratio", type=float, default=0.75)
    parser.add_argument("--strengthen_lambda", action="store_true")

    parser.add_argument("--do_xentropy", action="store_true")
    parser.add_argument("--xentropy_sparse_loss_weight", type=float, default=1.0)
    parser.add_argument("--xentropy_temperature", type=float, default=1.0)

    parser.add_argument("--distill_reranker", type=str, default=None)
    parser.add_argument("--distill_reranker_margin", type=float, default=None)
    parser.add_argument("--distill_reranker_tau", type=float, default=1.0)
    parser.add_argument("--distill_reranker2dense_loss_weight", type=float, default=1.0)

    parser.add_argument("--disable_autocast", action="store_true")


class LexmaeLearner(LearnerMixin):
    def __init__(self, model_args, config, tokenizer, encoder, query_encoder=None):
        super().__init__(model_args, config, tokenizer, encoder, query_encoder, )

        self.sim_fct = Similarity(metric="dot")
        self.flops_loss = FLOPS()

        self.reranker = None
        if model_args.distill_reranker:
            reranker_config = AutoConfig.from_pretrained(
                model_args.distill_reranker)
            self.reranker = BertForSequenceClassification.from_pretrained(
                model_args.distill_reranker, config=reranker_config)

    def forward(
            self,
            input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            input_ids_query=None, attention_mask_query=None, token_type_ids_query=None, position_ids_query=None,
            distill_labels=None, training_progress=None,
            training_mode=None,
            **kwargs,
    ):

        if training_mode is None:
            return self.encoder(input_ids, attention_mask, ).contiguous()

        dict_for_meta = {}

        # input reshape
        (high_dim_flag, num_docs), (org_doc_shape, org_query_shape), \
        (input_ids, attention_mask, token_type_ids, position_ids,), \
        (input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,) = preproc_inputs(
            input_ids, attention_mask, token_type_ids, position_ids,
            input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,)
        bsz = org_doc_shape[0]

        # freezed cross-encoder before main netowrk to save memory
        if self.model_args.distill_reranker is not None:
            with torch.no_grad():
                cross_input_ids, cross_attention_mask, cross_token_type_ids = combine_dual_inputs_by_attention_mask(
                    input_ids_query.unsqueeze(1).expand(-1, num_docs, -1).contiguous().view(bsz * num_docs, -1),
                    attention_mask_query.unsqueeze(1).expand(-1, num_docs, -1).contiguous().view(bsz * num_docs, -1),
                    input_ids, attention_mask, )
                reranker_outputs = self.reranker(
                    cross_input_ids, cross_attention_mask, cross_token_type_ids,
                    output_attentions=True, output_hidden_states=True)
                reranker_scores = reranker_outputs.logits.squeeze(-1).view(bsz, num_docs)

        # encoding
        doc_outputs = self.encoding_doc(input_ids, attention_mask, return_dict=True)
        query_outputs = self.encoding_query(input_ids_query, attention_mask_query, return_dict=True)

        # emb_doc, emb_query = doc_outputs["sentence_embedding"].to(torch.float32), query_outputs["sentence_embedding"].to(torch.float32)
        emb_doc, emb_query = doc_outputs["sentence_embedding"], query_outputs["sentence_embedding"]

        dict_for_loss = {}

        # sparsity
        # sparsity loss
        with torch.cuda.amp.autocast(enabled=not self.model_args.disable_autocast):
            if self.model_args.disable_autocast:
                emb_doc, emb_query = emb_doc.to(torch.float32), emb_query.to(torch.float32)

            self.calc_sparse_regularization(
                emb_query, emb_doc, dict_for_meta, dict_for_loss, num_docs, )
            self.log_sparse_info(
                emb_query, emb_doc, dict_for_meta, query_mask=attention_mask_query, doc_mask=attention_mask,
                query_ids=input_ids_query, doc_ids=input_ids,)

            sim_out_mask = None
            if self.model_args.distill_reranker is not None:
                sparse_nb_similarities = self.calc_sims_without_inbatch(emb_query, emb_doc, num_docs) / self.model_args.xentropy_temperature
                self.calc_distill_loss(
                    sparse_nb_similarities, reranker_scores, dict_for_meta, dict_for_loss,
                    source_name="sparse", target_name="reranker", sim_neg_mask=None,
                    tau=self.model_args.distill_reranker_tau, )
                if self.model_args.distill_reranker_margin is not None and self.model_args.distill_reranker_margin > -99:
                    bi_similarities_mask = self.generate_bi_similarities_mask_by_reranker(
                        reranker_scores, do_in_batch=True, logits_margin=self.model_args.distill_reranker_margin)
                    sim_out_mask = 1 - bi_similarities_mask

            if self.model_args.do_xentropy:
                sparse_ib_similarities = self.calc_similarities(emb_query, emb_doc) / self.model_args.xentropy_temperature
                xentropy_target = torch.arange(bsz, device=input_ids.device, dtype=torch.long) * num_docs + \
                                  self.get_delta(bsz, num_docs)
                self.calc_xentropy_loss_for_sims(
                    sparse_ib_similarities, xentropy_target, dict_for_meta, dict_for_loss,
                    loss_name="sparse", sim_out_mask=sim_out_mask)

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

    def calc_sims_without_inbatch(self, emb_query, emb_doc, num_docs):
        emb_dim = emb_doc.shape[-1]
        emb_doc = emb_doc.view(-1, num_docs, emb_dim)
        emb_query = emb_query.unsqueeze(1)  # [bs, 1, emb_dim]
        return torch.sum(emb_doc * emb_query, dim=-1)  # [ns, nd]

    def calc_similarities(self,  emb_query, emb_doc):
        emb_dim = emb_doc.shape[-1]
        ga_emb_doc = self.gather_tensor(emb_doc)
        similarities = self.sim_fct(emb_query, ga_emb_doc.view(-1, emb_dim))
        return similarities

    def calc_xentropy_loss_for_sims(
            self, similarities, target, dict_for_meta, dict_for_loss,
            loss_name="dense", sim_out_mask=None):
        dict_for_meta[f"xentropy_{loss_name}_loss_weight"] = getattr(
            self.model_args, f"xentropy_{loss_name}_loss_weight", 1.0)
        if sim_out_mask is not None:
            proc_similarities = self.mask_out_logits(similarities, sim_out_mask)
            with torch.no_grad():  # logging
                dict_for_meta[f"xentropy_{loss_name}_valid_ratio"] = \
                    (1 - sim_out_mask).to(similarities.dtype).mean(dim=-1).mean().detach().item()
        else:
            proc_similarities = similarities
        dict_for_loss[f"xentropy_{loss_name}_loss"] = nn.CrossEntropyLoss()(proc_similarities, target)

    def log_sparse_info(
            self, emb_query, emb_doc, dict_for_meta,
            query_mask=None, doc_mask=None,
            query_ids=None, doc_ids=None,
    ):
        with torch.no_grad():
            emb_dim = emb_doc.shape[-1]
            emb_doc_detach = emb_doc.view(-1, emb_dim).detach()
            dict_for_meta["log_sparse_doc_dup_tks"] = int(emb_doc_detach.sum(-1).mean().cpu().item() * 100)
            doc_idt_tks = (emb_doc_detach > 0.).to(emb_doc_detach.dtype).sum(-1)
            dict_for_meta["log_sparse_doc_idt_tks"] = doc_idt_tks.mean().cpu().item()
            if doc_mask is not None:
                doc_lens = doc_mask.view(emb_doc_detach.shape[0], -1).sum(-1)
                dict_for_meta["log_sparse_doc_ratio"] = (doc_idt_tks / (doc_lens.to(emb_doc_detach.dtype) + 1e-4)).mean().cpu().item()
                if doc_ids is not None:
                    ratio_list = []
                    for ids, dl, num_idt_tks in zip(
                            doc_ids.cpu().numpy(), doc_lens.cpu().numpy(), doc_idt_tks.cpu().numpy()):
                        ratio_list.append(float(num_idt_tks * 1.0 / (len(set(ids[1:dl - 1])) + 1e-4)))
                    dict_for_meta["log_sparse_doc_ratio_exact"] = sum(ratio_list) / len(ratio_list)

            emb_query_detach = emb_query.view(-1, emb_dim).detach()
            dict_for_meta["log_sparse_query_dup_tks"] = int(emb_query_detach.sum(-1).mean().cpu().item() * 100)
            query_idt_tks = (emb_query_detach > 0.).to(emb_query_detach.dtype).sum(-1)
            dict_for_meta["log_sparse_query_idt_tks"] = query_idt_tks.mean().cpu().item()
            if query_mask is not None:
                query_lens = query_mask.view(emb_query_detach.shape[0], -1).sum(-1)
                dict_for_meta["log_sparse_query_ratio"] = (query_idt_tks / (query_lens.to(emb_query_detach.dtype) + 1e-4)).mean().cpu().item()
                if query_ids is not None:
                    ratio_list = []
                    for ids, dl, num_idt_tks in zip(
                            query_ids.cpu().numpy(), query_lens.cpu().numpy(), query_idt_tks.cpu().numpy()):
                        ratio_list.append(float(num_idt_tks * 1.0 / (len(set(ids[1:dl - 1])) + 1e-4)))
                    dict_for_meta["log_sparse_query_ratio_exact"] = sum(ratio_list) / len(ratio_list)

    def calc_sparse_regularization(
            self, emb_query, emb_doc,
            dict_for_meta, dict_for_loss, num_docs,
    ):
        emb_dim = emb_doc.shape[-1]

        dict_for_meta["flops_doc_loss_weight"] = self.model_args.lambda_d
        dict_for_meta["flops_query_loss_weight"] = self.model_args.lambda_q \
            if self.model_args.lambda_q is not None else self.model_args.lambda_d * self.model_args.lambda_ratio

        if self.model_args.strengthen_lambda:
            emb_doc_rsp = emb_doc.view(-1, num_docs, emb_dim)
            pos_emb_doc, neg_emb_doc = emb_doc_rsp[:, 0], emb_doc_rsp[:, 1]
            dict_for_loss["flops_doc_loss"] = self.flops_loss(pos_emb_doc) + self.flops_loss(neg_emb_doc)
        else:
            dict_for_loss["flops_doc_loss"] = self.flops_loss(emb_doc)
        dict_for_loss["flops_query_loss"] = self.flops_loss(emb_query)

    def calc_distill_loss(
            self, source_similarities, target_similarities,
            dict_for_meta=None, dict_for_loss=None, source_name=None,
            target_name=None, sim_neg_mask=None, tau=1.0,
    ):
        kldiv_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

        if sim_neg_mask is not None:
            source_similarities = self.mask_out_logits(source_similarities / tau, sim_neg_mask)
            target_similarities = self.mask_out_logits(target_similarities / tau, sim_neg_mask)

        source_logp = torch.log_softmax(source_similarities, dim=-1)
        target_prob = torch.softmax(target_similarities, dim=-1)

        loss = kldiv_fct(source_logp, target_prob)

        if source_name is not None and target_name is not None:
            loss_str = f"distill_{target_name}2{source_name}_loss"
            dict_for_meta[f"{loss_str}_weight"] = getattr(self.model_args, f"{loss_str}_weight", 1.0) * (tau ** 2)
            dict_for_loss[loss_str] = loss
        return loss

    def generate_bi_similarities_mask_by_reranker(
            self, reranker_scores, do_in_batch=False, logits_margin=1.0):
        # [bs, *], [bs, num_docs]
        with torch.no_grad():
            bsz, num_docs = reranker_scores.shape
            positive_scores = reranker_scores[:, :1]  # [bs,1]
            margin_tn_mask = (reranker_scores[:, 1:] < (positive_scores - logits_margin)).to(torch.long)
            margin_mask = torch.cat(
                [torch.ones_like(positive_scores, dtype=torch.long), margin_tn_mask], dim=1).contiguous()

            if do_in_batch:
                device = reranker_scores.device
                anchors = torch.arange(bsz, device=device, dtype=torch.long) * num_docs + self.get_delta(bsz, num_docs)  # [bsz]
                idxs = anchors.unsqueeze(1) + torch.arange(num_docs, device=device, dtype=torch.long).unsqueeze(0) # [bsz,num_docs]
                ib_margin_mask = torch.ones([bsz, self.my_world_size * bsz * num_docs], device=device, dtype=torch.long)
                ib_margin_mask = torch.scatter(ib_margin_mask, dim=1, index=idxs, src=margin_mask)
                return ib_margin_mask
            else:
                return margin_mask

    def get_delta(self, bsz, num_docs):
        return dist.get_rank() * bsz * num_docs if self.my_world_size > 1 else 0


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
            sync_context = model.no_sync if accelerator.distributed_type != accelerate.DistributedType.NO and \
                                            step % args.gradient_accumulation_steps > 0 else nullcontext

            with sync_context():  # disable DDP sync for accumulation step
                outputs = model(training_mode="retrieval_finetune", **batch)
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
                        args, accelerator, args.output_dir, model, tokenizer, args, save_specified_module="encoder",)

                # DDP for eval
                if eval_fn is not None and args.eval_steps > 0 and (
                        global_step % args.eval_steps == 0 or global_step == args.max_train_steps-1):
                    key_metric, eval_metrics = eval_fn(
                        args, eval_dataset, model, accelerator, global_step=global_step,
                        tb_writer=tb_writer, tokenizer=tokenizer, key_metric_name="MRR@10",
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
            save_specified_module="encoder",)
    model.zero_grad()
    model.eval()
    accelerator.wait_for_everyone()

def custom_hard_negative_preparation(args, ):
    assert args.negs_sources.startswith("custom")
    assert args.negs_source_paths is not None

    all_group_names = ["dense", "sparse", "uni"]
    group_name_to_index = {"dense": 0, "sparse": 1, "uni": 2}
    if args.split_negs:
        str_sources = args.negs_source_paths.strip("|").split("|")
        qid2negatives = dict()
        for str_source in str_sources:
            if str_source.startswith("dense:"):
                group_name = "dense"
                paths = str_source[6:].strip(";").split(";")
            elif str_source.startswith("sparse:"):
                group_name = "sparse"
                paths = str_source[7:].strip(";").split(";")
            elif str_source.startswith("uni:"):
                group_name = "uni"
                paths = str_source[7:].strip(";").split(";")
            else:
                raise AttributeError(str_sources, str_source)

            for pkl_path in paths:
                local_qid2negatives = load_pickle(pkl_path)
                for qid, lc_negs in local_qid2negatives.items():
                    qid = int(qid)
                    if qid not in qid2negatives:
                        qid2negatives[qid] = [(gn, []) for gn in all_group_names] if args.split_negs else []
                    if args.split_negs: #
                        qid2negatives[qid][group_name_to_index[group_name]][1].extend(lc_negs[:args.num_negs_per_system])
                    else:
                        qid2negatives[qid].extend(lc_negs[:args.num_negs_per_system])
        # got qid2negatives!
    else:  # default
        qid2negatives = dict()
        neg_filepath_list = args.negs_source_paths.strip(";").split(";")
        for neg_filepath in neg_filepath_list:
            for qid, neg_pids in load_pickle(neg_filepath).items():
                if qid not in qid2negatives:
                    qid2negatives[qid] = []
                qid2negatives[qid].extend(neg_pids[:args.num_negs_per_system])
    return qid2negatives


def main():
    parser = argparse.ArgumentParser()
    # add task specific hyparam
    #
    define_hparams_training(parser)

    parser.add_argument("--data_load_type", type=str, default="disk", choices=["disk", "memory"])
    parser.add_argument("--data_dir", type=str,
                        default=USER_HOME + "/ws/data/set/")  # princeton-nlp/sup-simcse-bert-base-uncased
    parser.add_argument("--num_negatives", type=int, default=7)
    parser.add_argument("--num_dev", type=int, default=500)
    parser.add_argument("--dev_type", type=str, default="dev", )

    parser.add_argument("--ce_score_margin", type=float, default=3.0)
    parser.add_argument("--num_negs_per_system", type=int, default=8)
    parser.add_argument("--negs_sources", type=str, default=None)
    parser.add_argument("--negs_source_paths", type=str, default=None)
    parser.add_argument("--split_negs", action="store_true")

    parser.add_argument("--no_title", action="store_true")
    parser.add_argument("--encoder", type=str, default="distilbert", )

    parser.add_argument("--eval_reranking_source", type=str, default=None)
    parser.add_argument("--prediction_source", type=str, default="dev", )

    parser.add_argument("--hits_num", type=int, default=1000)

    # for hard negative sampling
    parser.add_argument("--anserini_path", type=str, default=None)

    parser.add_argument("--do_hn_gen", action="store_true")  # hard negative
    parser.add_argument("--hn_gen_num", type=int, default=1000)

    model_param_list = add_model_hyperparameters(parser)
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

    if args.encoder == "distilbert":
        encoder_class = DistilBertSpladeEnocder
    elif args.encoder == "bert":
        encoder_class = BertSpladeEnocder
    elif args.encoder == "roberta":
        encoder_class = RobertaSpladeEnocder
    else:
        raise NotImplementedError(args.encoder)

    encoder = encoder_class.from_pretrained(args.model_name_or_path, config=config)
    embedding_dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else len(tokenizer.get_vocab())
    model = encoder

    if args.do_train:
        model = LexmaeLearner(args, config, tokenizer, encoder, query_encoder=None)
        with accelerator.main_process_first():
            train_dataset = DatasetMarcoPassagesRanking(
                "train", args.data_dir, args.data_load_type, args, tokenizer, add_title=(not args.no_title))
        with accelerator.main_process_first():
            if args.eval_reranking_source is None:
                dev_dataset = DatasetRerank(
                    "dev", args.data_dir, "memory", args, tokenizer, num_dev=args.num_dev, add_title=(not args.no_title))
            else:
                dev_dataset = DatasetCustomRerank(
                    args.dev_type, args.data_dir, "memory", args, tokenizer, num_dev=args.num_dev, add_title=(not args.no_title),
                    filepath_dev_qid2top1000pids=args.eval_reranking_source,
                )

        if args.negs_sources == "official":
            train_dataset.load_official_bm25_negatives(keep_num_neg=args.num_negs_per_system, )
        elif args.negs_sources.startswith("custom"):
            qid2negatives = custom_hard_negative_preparation(args)
            train_dataset.use_new_qid2negatives(qid2negatives, accelerator=None)
        else:
            train_dataset.load_sbert_hard_negatives(
                ce_score_margin=args.ce_score_margin,
                num_negs_per_system=args.num_negs_per_system,
                negs_sources=args.negs_sources)
        train(
            args, train_dataset, model, accelerator, tokenizer,
            eval_dataset=dev_dataset, eval_fn=evaluate_encoder_reranking)

    if args.do_eval or args.do_prediction or args.do_hn_gen:
        if args.do_train:
            encoder = encoder_class.from_pretrained(pretrained_model_name_or_path=args.output_dir, config=config)
        else:
            encoder = model
        encoder = accelerator.prepare(encoder)

        meta_best_str = ""
        if args.do_eval:
            if args.eval_reranking_source is None:
                dev_dataset = DatasetRerank(
                    "dev", args.data_dir, "memory", args, tokenizer, num_dev=None, add_title=(not args.no_title))
            else:
                dev_dataset = DatasetCustomRerank(
                    args.dev_type, args.data_dir, "memory", args, tokenizer, num_dev=None,
                    add_title=(not args.no_title),
                    filepath_dev_qid2top1000pids=args.eval_reranking_source,
                )

            best_dev_result, best_dev_metric = evaluate_encoder_reranking(
                args, dev_dataset, encoder, accelerator, global_step=None,
                save_prediction=True, tokenizer=tokenizer, key_metric_name="MRR@10",
                similarity_metric=None, query_model=None,)
            if accelerator.is_local_main_process:
                # meta_best_str += f"best_test_result: {best_dev_result}, "
                meta_best_str += json.dumps(best_dev_metric) + os.linesep
        else:
            best_dev_result = None

        # cannot do_prediction and do_hn_gen in one script!!!
        assert not (args.do_prediction and args.do_hn_gen), "cannot do_prediction and do_hn_gen in one script!!!"

        if args.do_prediction:
            best_pred_result, dev_pred_metric = evaluate_sparse_retreival(
                args, args.prediction_source, encoder, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                key_metric_name="MRR@10", delete_model=False, add_title=(not args.no_title), query_model=None,
                # vocab_id2token=vocab_id2token,
                tokenizer=tokenizer, quantization_factor=100, anserini_path=args.anserini_path,
                hits=args.hits_num,
            )
            # meta_best_str += json.dumps(dev_pred_metric) + os.linesep

        if accelerator.is_local_main_process:
            with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
                fp.write(f"{best_dev_result}, {meta_best_str}")

        if args.do_hn_gen:
            get_hard_negative_by_sparse_retrieval(
                args, None, encoder, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                key_metric_name="MRR@10", delete_model=False, add_title=(not args.no_title), query_model=None,
                # vocab_id2token=vocab_id2token,
                tokenizer=tokenizer, quantization_factor=100, anserini_path=args.anserini_path,
                hits=args.hn_gen_num,
            )

if __name__ == '__main__':
    main()

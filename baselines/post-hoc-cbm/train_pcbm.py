import argparse
import os
import pickle
import re
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    return parser.parse_args()


def sanitize_filename(value):
    return re.sub(r'[<>:"/\\|?*]+', '_', str(value)).strip(' ._') or 'unknown'


def run_linear_probe(args, train_data, test_data):
    train_features, train_labels = train_data
    test_features, test_labels = test_data

    # 我们使用 SGDClassifier
    classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
                               alpha=args.lam, l1_ratio=args.alpha, verbose=0,
                               penalty="elasticnet", max_iter=10000)
    classifier.fit(train_features, train_labels)

    train_predictions = classifier.predict(train_features)
    train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
    predictions = classifier.predict(test_features)
    test_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

    # 计算类级准确率
    cls_acc = {"train": {}, "test": {}}
    for lbl in np.unique(train_labels):
        test_lbl_mask = test_labels == lbl
        train_lbl_mask = train_labels == lbl
        cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == predictions[test_lbl_mask]).astype(float))
        cls_acc["train"][lbl] = np.mean(
            (train_labels[train_lbl_mask] == train_predictions[train_lbl_mask]).astype(float))
        print(f"{lbl}: {cls_acc['test'][lbl]}")

    run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
                "cls_acc": cls_acc,
                }

    # 如果是二分类任务，计算 AUC
    if test_labels.max() == 1:
        run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))
    return run_info, classifier.coef_, classifier.intercept_


def main(args, concept_bank, backbone, preprocess):
    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)

    # 将目标任务、骨干和概念库来源一起写入文件名，避免不同实验相互覆盖
    conceptbank_source = sanitize_filename(os.path.splitext(os.path.basename(args.concept_bank))[0])
    dataset_name = sanitize_filename(args.dataset)
    backbone_name = sanitize_filename(args.backbone_name)
    num_classes = len(classes)

    # 初始化 PCBM 模块
    posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class,
                                     n_classes=num_classes)
    posthoc_layer = posthoc_layer.to(args.device)

    # 计算投影并保存到输出目录
    train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(args, backbone,
                                                                                                        posthoc_layer,
                                                                                                        train_loader,
                                                                                                        test_loader)

    run_info, weights, bias = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls))
    run_info["config"] = {
        "dataset": args.dataset,
        "backbone_name": args.backbone_name,
        "concept_bank": args.concept_bank,
        "lam": args.lam,
        "alpha": args.alpha,
        "seed": args.seed,
        "device": args.device,
    }

    # 从 SGDClassifier 模块转换到 PCBM 模块
    posthoc_layer.set_weights(weights=weights, bias=bias)

    # 修改：生成不冲突的模型保存路径
    model_path = os.path.join(args.out_dir,
                              f"pcbm_{dataset_name}__{backbone_name}__{conceptbank_source}.ckpt")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save(posthoc_layer, model_path)

    # 保存运行信息
    run_info_file = model_path.replace("pcbm", "run_info-pcbm")
    run_info_file = run_info_file.replace(".ckpt", ".pkl")

    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)

    if num_classes > 1:
        # 打印每个类的前5个概念权重
        print(posthoc_layer.analyze_classifier(k=5))

    print(f"模型保存到: {model_path}")
    print(run_info)


if __name__ == "__main__":
    args = config()

    # 加载概念库
    with open(args.concept_bank, 'rb') as f:
        all_concepts = pickle.load(f)
    all_concept_names = list(all_concepts.keys())
    print(f"概念库路径: {args.concept_bank}. 将使用 {len(all_concept_names)} 个概念.")
    concept_bank = ConceptBank(all_concepts, args.device)

    # 从模型库获取骨干网络
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, concept_bank, backbone, preprocess)
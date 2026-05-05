import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from .constants import (
    AWA2_CLASSES_FILE,
    AWA2_DATA_DIR,
    AWA2_IMAGE_CLASS_LABELS_FILE,
    AWA2_IMAGE_DIR,
    AWA2_IMAGES_FILE,
    AWA2_PREDICATE_MATRIX_FILE,
    AWA2_PREDICATES_FILE,
    AWA2_TEST_CLASSES_FILE,
    AWA2_TRAIN_CLASSES_FILE,
    AWA2_TRAIN_TEST_SPLIT_FILE,
)


class AwA2Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataframe.iloc[idx]
        image = Image.open(row["img_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["class_idx"])


class AwA2ConceptDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def _read_nonempty_lines(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        return [line.strip() for line in handle if line.strip()]


def _parse_indexed_names(file_path):
    names = []
    for line in _read_nonempty_lines(file_path):
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            names.append(parts[1].strip())
        else:
            names.append(line.strip())
    return names


def _resolve_name(entry, names):
    entry = str(entry).strip()
    if not entry:
        return None
    parts = entry.split(maxsplit=1)
    if parts[0].isdigit():
        index = int(parts[0]) - 1
        if 0 <= index < len(names):
            return names[index]
    if entry in names:
        return entry
    if len(parts) == 2 and parts[1].strip() in names:
        return parts[1].strip()
    return entry


def _build_image_lookup():
    lookup = {}
    candidate_roots = [AWA2_IMAGE_DIR, AWA2_DATA_DIR]
    for root in candidate_roots:
        if not os.path.isdir(root):
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for image_path in glob(os.path.join(root, "**", ext), recursive=True):
                normalized_path = os.path.normpath(image_path).lower()
                lookup.setdefault(normalized_path, image_path)
                lookup.setdefault(os.path.basename(image_path).lower(), image_path)
                rel_path = os.path.relpath(image_path, root).replace("\\", "/").lower()
                lookup.setdefault(rel_path, image_path)
    return lookup


def _resolve_image_path(image_lookup, image_name):
    normalized_name = str(image_name).replace("\\", "/").strip().lower()
    candidates = [
        normalized_name,
        os.path.normpath(str(image_name)).lower(),
        os.path.basename(str(image_name)).lower(),
    ]
    for candidate in candidates:
        if candidate in image_lookup:
            return image_lookup[candidate]
    basename = os.path.basename(str(image_name))
    if os.path.isdir(AWA2_IMAGE_DIR):
        matches = glob(os.path.join(AWA2_IMAGE_DIR, "**", basename), recursive=True)
        if len(matches) == 1:
            return matches[0]
    matches = glob(os.path.join(AWA2_DATA_DIR, "**", basename), recursive=True)
    if len(matches) == 1:
        return matches[0]
    return None


def _load_awa2_frame():
    class_names = _parse_indexed_names(AWA2_CLASSES_FILE)
    attr_names = _parse_indexed_names(AWA2_PREDICATES_FILE)
    attr_matrix = np.loadtxt(AWA2_PREDICATE_MATRIX_FILE)
    if attr_matrix.ndim == 1:
        attr_matrix = attr_matrix.reshape(1, -1)

    image_lookup = _build_image_lookup()

    if os.path.exists(AWA2_IMAGES_FILE) and os.path.exists(AWA2_IMAGE_CLASS_LABELS_FILE):
        images_df = pd.read_csv(
            AWA2_IMAGES_FILE,
            sep=r"\s+",
            header=None,
            names=["image_id", "image_name"],
            engine="python",
        )
        labels_df = pd.read_csv(
            AWA2_IMAGE_CLASS_LABELS_FILE,
            sep=r"\s+",
            header=None,
            names=["image_id", "class_idx"],
            engine="python",
        )
        frame = images_df.merge(labels_df, on="image_id", how="inner")
        if os.path.exists(AWA2_TRAIN_TEST_SPLIT_FILE):
            split_df = pd.read_csv(
                AWA2_TRAIN_TEST_SPLIT_FILE,
                sep=r"\s+",
                header=None,
                names=["image_id", "is_train"],
                engine="python",
            )
            frame = frame.merge(split_df, on="image_id", how="inner")
        frame["class_idx"] = frame["class_idx"].astype(int) - 1
        frame["class_name"] = frame["class_idx"].map(lambda x: class_names[x] if 0 <= x < len(class_names) else str(x))
        frame["img_path"] = frame["image_name"].map(lambda name: _resolve_image_path(image_lookup, name))
        frame = frame[frame["img_path"].notna()].copy()
    else:
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(glob(os.path.join(AWA2_IMAGE_DIR, "**", ext), recursive=True))
        if not image_paths:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                image_paths.extend(glob(os.path.join(AWA2_DATA_DIR, "**", ext), recursive=True))

        records = []
        base_root = AWA2_IMAGE_DIR if os.path.isdir(AWA2_IMAGE_DIR) else AWA2_DATA_DIR
        for image_path in image_paths:
            rel_path = os.path.relpath(image_path, base_root).replace("\\", "/")
            class_name = rel_path.split("/")[0] if "/" in rel_path else os.path.basename(os.path.dirname(image_path))
            if class_name not in class_names:
                continue
            records.append(
                {
                    "image_id": len(records),
                    "image_name": rel_path,
                    "class_idx": class_names.index(class_name),
                    "class_name": class_name,
                    "img_path": image_path,
                }
            )
        frame = pd.DataFrame.from_records(records)

    if frame.empty:
        raise ValueError("No AwA2 images were found. Check AWA2_DATA_DIR and image layout.")

    # AI-added for AwA2: ignore the original seen/unseen split and use a standard 8:2
    # stratified image split so all 50 classes participate in training and testing.
    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=frame["class_idx"],
    )
    train_frame = train_frame.copy().reset_index(drop=True)
    test_frame = test_frame.copy().reset_index(drop=True)

    idx_to_class = {idx: name for idx, name in enumerate(class_names)}
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}

    # normalize class_idx columns
    train_frame["class_idx"] = train_frame["class_idx"].astype(int)
    test_frame["class_idx"] = test_frame["class_idx"].astype(int)

    return {
        "frame": frame,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "attr_names": attr_names,
        "attr_matrix": attr_matrix,
    }


def load_awa2_data(args, preprocess):
    from torchvision import transforms

    metadata = _load_awa2_frame()
    train_frame = metadata["train_frame"]
    test_frame = metadata["test_frame"]

    # AI-added for AwA2: remove stale cached projection files so downstream code
    # recomputes embeddings/projections using the new 8:2 stratified split.
    try:
        out_dir = getattr(args, 'out_dir', None) or os.path.join(os.getcwd(), 'conceptbanks')
        patterns = [
            'train-embs_awa2*', 'test-embs_awa2*', 'train-proj_awa2*', 'test-proj_awa2*',
            'train-lbls_awa2*', 'test-lbls_awa2*'
        ]
        for pat in patterns:
            for p in glob(os.path.join(out_dir, pat)):
                try:
                    os.remove(p)
                    print(f"[AWA2-loader] removed cached projection file: {p}")
                except Exception:
                    pass
    except Exception:
        pass

    if preprocess is None:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )
    else:
        train_transform = preprocess
        test_transform = preprocess

    train_dataset = AwA2Dataset(train_frame, transform=train_transform)
    test_dataset = AwA2Dataset(test_frame, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, test_loader, metadata["idx_to_class"], metadata["class_names"]


def awa2_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed):
    metadata = _load_awa2_frame()
    frame = metadata["train_frame"] if not metadata["train_frame"].empty else metadata["frame"]
    class_names = metadata["class_names"]
    attr_names = metadata["attr_names"]
    attr_matrix = metadata["attr_matrix"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    concept_loaders = {}
    for attr_idx in range(attr_matrix.shape[1]):
        concept_name = attr_names[attr_idx] if attr_idx < len(attr_names) else f"attr_{attr_idx}"
        positive_classes = {class_names[class_idx] for class_idx in np.where(attr_matrix[:, attr_idx] > 0)[0]}
        negative_classes = set(class_names) - positive_classes

        pos_df = frame[frame["class_name"].isin(positive_classes)].copy()
        neg_df = frame[frame["class_name"].isin(negative_classes)].copy()

        if pos_df.empty or neg_df.empty:
            fallback_frame = metadata["frame"]
            pos_df = fallback_frame[fallback_frame["class_name"].isin(positive_classes)].copy()
            neg_df = fallback_frame[fallback_frame["class_name"].isin(negative_classes)].copy()

        if pos_df.empty or neg_df.empty:
            print(f"Skipping AwA2 concept {concept_name}: insufficient positive/negative samples.")
            continue

        if pos_df.shape[0] < 2 * n_samples:
            pos_df = pos_df.sample(2 * n_samples, replace=True, random_state=seed)
        else:
            pos_df = pos_df.sample(2 * n_samples, replace=False, random_state=seed)

        if neg_df.shape[0] < 2 * n_samples:
            neg_df = neg_df.sample(2 * n_samples, replace=True, random_state=seed)
        else:
            neg_df = neg_df.sample(2 * n_samples, replace=False, random_state=seed)

        pos_ds = AwA2ConceptDataset(pos_df["img_path"].tolist(), transform=preprocess)
        neg_ds = AwA2ConceptDataset(neg_df["img_path"].tolist(), transform=preprocess)
        pos_loader = DataLoader(pos_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        neg_loader = DataLoader(neg_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        concept_loaders[concept_name] = {"pos": pos_loader, "neg": neg_loader}

    return concept_loaders
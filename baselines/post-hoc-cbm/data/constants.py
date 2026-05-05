import os

# CUB Constants
# CUB data is downloaded from the CBM release.
# Dataset: https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/ 
# Metadata and splits: https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683
CUB_DATA_DIR = "D:\Edge-download\CUB_200_2011"
CUB_PROCESSED_DIR = "D:\Edge-download\CUB_processed\class_attr_data_10"


# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DERM7_FOLDER = "/path/to/derm7pt/"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")

# Ham10000 can be obtained from : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
HAM10K_DATA_DIR = "D:\Agent_CBM\Datasets\HAM10000"


# BRODEN concept bank
BRODEN_CONCEPTS = "D:\Agent_CBM\Datasets\CIFAR_concepts-Broaden\\broden_concepts"

# AwA2 data constants
AWA2_DATA_DIR = r"D:\Agent_CBM\Datasets\AWA2\AwA2-data\Animals_with_Attributes2"
AWA2_IMAGE_DIR = os.path.join(AWA2_DATA_DIR, "JPEGImages")
AWA2_CLASSES_FILE = os.path.join(AWA2_DATA_DIR, "classes.txt")
AWA2_PREDICATES_FILE = os.path.join(AWA2_DATA_DIR, "predicates.txt")
AWA2_PREDICATE_MATRIX_FILE = os.path.join(AWA2_DATA_DIR, "predicate-matrix-binary.txt")
AWA2_TRAIN_CLASSES_FILE = os.path.join(AWA2_DATA_DIR, "trainclasses.txt")
AWA2_TEST_CLASSES_FILE = os.path.join(AWA2_DATA_DIR, "testclasses.txt")
AWA2_TRAIN_TEST_SPLIT_FILE = os.path.join(AWA2_DATA_DIR, "train_test_split.txt")
AWA2_IMAGES_FILE = os.path.join(AWA2_DATA_DIR, "images.txt")
AWA2_IMAGE_CLASS_LABELS_FILE = os.path.join(AWA2_DATA_DIR, "image_class_labels.txt")
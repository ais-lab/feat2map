import argparse
import subprocess
from pathlib import Path
import os
from pathlib import Path
from pprint import pformat
from hloc import (
    extract_features,
    match_features,
    pairs_from_covisibility,
    triangulation
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.read_write_model import read_model
import random
from run_custom_data.directSfM_pipeline import direct_sfm


def re_triangulation(args, colmap_sfm="colmap_sift/sparse/0", test_rate=0.1):
    
    outputs = args.workspace_path
    images = outputs / "images"
    sfm_pairs = outputs / "pairs-db-covis20.txt"  # top 20 most covisible in SIFT model
    reference_sfm = outputs / "sfm_superpoint+superglue"  # the SfM model we will build

    # list the standard configurations available
    print(f"Configs for feature extractors:\n{pformat(extract_features.confs)}")
    print(f"Configs for feature matchers:\n{pformat(match_features.confs)}")

    # pick one of the configurations for image retrieval, local feature extraction, and matching
    # you can also simply write your own here!
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    # extract local features
    features = extract_features.main(feature_conf, images, outputs)
    
    pairs_from_covisibility.main(outputs / colmap_sfm, sfm_pairs, num_matched=20)
    
    sfm_matches = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    
    reconstruction = triangulation.main(
    reference_sfm, outputs / colmap_sfm, images, sfm_pairs, features, sfm_matches
    )
    create_test_list(reference_sfm, args.workspace_path, test_rate=0.1)
    print(f"Randomly selecte {test_rate} test images and save in 'test_list.txt'")

def run_colmap(confs):
    workspace_path = confs.workspace_path
    images_path = workspace_path / "images"

    assert workspace_path.exists(), workspace_path
    assert images_path.exists(), images_path

    # mkdir colmap_sift
    workspace_path =  workspace_path / "colmap_sift"
    workspace_path.mkdir(exist_ok=True, parents=True)
    
    colmap_command = ["colmap", "automatic_reconstructor"]
    args = {
            "--workspace_path": workspace_path,
            "--image_path": images_path,
            "--dense": "0"
            }
    for k, v in args.items():
        colmap_command.append(str(k))
        colmap_command.append(str(v))

    print("Reconstructing 3D model")
    process = subprocess.Popen(colmap_command, 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
    stdout, stderr = process.communicate()

    print(process.returncode)
    print(stdout)
    print(stderr)
    print("3D model reconstruction done")

def create_test_list(reference_sfm, workspace_path, test_rate=0.1):
    _, images_train, _ = read_model(reference_sfm, ext=".bin")
    list_images = [image.name for i, image in images_train.items()]
    # shuffle the list
    random.shuffle(list_images)
    # split the list
    test_list = list_images[:int(len(list_images)*test_rate)]
    with open(workspace_path / "test_list.txt", "w") as f:
        for item in test_list:
            f.write("%s\n" % item)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace_path", default="../custom_dataset", type=Path)
    parser.add_argument("--use_colmap", action='store_true' , help='use pre-trained model')
    args = parser.parse_args()
    if args.use_colmap:
        print("use_colmap")
        # run_colmap(args)
        re_triangulation(args, colmap_sfm="sparse/0")
    else:
        direct_sfm(str(args.workspace_path))
        create_test_list(args.workspace_path / "sfm_superpoint+superglue/models", args.workspace_path, test_rate=0.1)
    
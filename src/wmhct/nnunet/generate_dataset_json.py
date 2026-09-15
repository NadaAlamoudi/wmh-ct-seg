#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""
Stage 4.1 - Write the nnU-Net dataset.json for a converted dataset.

Manuscript: Methods, segmentation model
Run: wmhct-dataset-json --help

Note: Derived from nnU-Net and remains under Apache-2.0.
"""

import argparse
import os
from typing import Tuple

import numpy as np

try:
    from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles
except ImportError as exc:  # pragma: no cover - depends on the optional extra
    raise ImportError(
        "This module needs nnU-Net, which is an optional dependency of wmhct because "
        "the rest of the pipeline and every statistical module runs without it.\n"
        "    pip install '.[nnunet]'\n"
        "batchgenerators is installed as part of nnunetv2."
    ) from exc


def get_identifiers_from_splitted_files(folder: str):
    return np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str,
                          modalities: Tuple, labels: dict, dataset_name: str,
                          sort_keys=True, license: str = "hands off!",
                          dataset_description: str = "", dataset_reference="",
                          dataset_release='0.0'):
    """
    :param output_file: full path to the dataset.json you intend to write
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: modality names, in the same order as the image channels
    :param labels: dict int->str; 0 must be background
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)
    test_identifiers = (get_identifiers_from_splitted_files(imagesTs_dir)
                        if imagesTs_dir is not None else [])

    json_dict = {
        'name': dataset_name,
        'description': dataset_description,
        'tensorImageSize': "4D",
        'reference': dataset_reference,
        'licence': license,
        'release': dataset_release,
        'modality': {str(i): modalities[i] for i in range(len(modalities))},
        'labels': {str(i): labels[i] for i in labels.keys()},
        'numTraining': len(train_identifiers),
        'numTest': len(test_identifiers),
        'training': [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
                     for i in train_identifiers],
        'test': ["./imagesTs/%s.nii.gz" % i for i in test_identifiers],
    }

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json. Proceeding anyway.")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)


def main():
    parser = argparse.ArgumentParser(description="Generate an nnU-Net dataset.json")
    parser.add_argument("--task_dir", required=True,
                        help="Task directory with imagesTr/, labelsTr/ and optionally imagesTs/")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--modality", default="CT",
                        help="Modality label for channel 0 (CT for the CT experiments, "
                             "MRI for the FLAIR ones)")
    args = parser.parse_args()

    imagesTr = os.path.join(args.task_dir, "imagesTr")
    imagesTs = os.path.join(args.task_dir, "imagesTs")
    if not os.path.isdir(imagesTs):
        imagesTs = None

    generate_dataset_json(
        output_file=os.path.join(args.task_dir, "dataset.json"),
        imagesTr_dir=imagesTr,
        imagesTs_dir=imagesTs,
        modalities={0: args.modality},
        labels={0: "background", 1: "WMH"},
        dataset_name=args.dataset_name,
    )
    print(f"dataset.json written to {args.task_dir}")


if __name__ == "__main__":
    main()

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple script for inspect checkpoint files."""
from tensorflow.python import pywrap_tensorflow


def print_tensors_in_checkpoint_file(file_name, tensor_name=None, all_tensors=True, all_tensor_names=True):
    """Prints tensors in a checkpoint file.

    If no `tensor_name` is provided, prints the tensor names and shape in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.

    Args:
        file_name: Name of the checkpoint file.
        tensor_name: Name of the tensor in the checkpoint file to print.
        all_tensors: Boolean indicating whether to print all tensors.
        all_tensor_names: Boolean indicating whether to print all tensor names.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors or all_tensor_names:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                print("tensor_name: ", key)
                if all_tensors:
                    print(reader.get_tensor(key))
        elif not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            print(reader.get_tensor(tensor_name))
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed with SNAPPY.")
        if "Data loss" in str(e) and (any([e in file_name for e in [".index", ".meta", ".data"]])):
            print("It's likely that this is a V2 checkpoint and you need to provide the filename prefix*. "
                  "Try removing the '.' and extension.")


if __name__ == "__main__":
    import os

    # Modify here to use your model file.
    model_file = os.path.join("C:\Python", "ChatLearner", "Data", "Result", "basic")
    # Modify here if you want to pass different parameters
    print_tensors_in_checkpoint_file(model_file)
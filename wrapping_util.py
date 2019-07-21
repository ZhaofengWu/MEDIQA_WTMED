import os
import sys

import torch

USAGE = 'Usage: python wrapping_util.py [wrap/unwrap] [path to model to be wrapped/unwrapped] [path to the mt_dnn .pt model as downloaded by the mt-dnn repo]'

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(USAGE)

    assert sys.argv[1] in ['wrap', 'unwrap'], USAGE
    model_path = sys.argv[2]
    assert os.path.exists(model_path), USAGE
    mtdnn_model_path = sys.argv[3]
    assert os.path.exists(mtdnn_model_path), USAGE
    output_path = 'new_pytorch_model.bin'
    assert not os.path.exists(output_path), 'There exists a `new_pytorch_model.bin`. Please first move/delete it'

    if sys.argv[1] == 'wrap':
        model = torch.load(model_path)
        mtdnn_model = torch.load(mtdnn_model_path)
        model = {'state': model, 'config': mtdnn_model['config']}
        torch.save(model, output_path)
    elif sys.argv[1] == 'unwrap':
        model = torch.load(model_path)
        model = model['state']
        torch.save(model, output_path)

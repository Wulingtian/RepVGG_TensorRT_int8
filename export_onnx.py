import onnx
import torch
from repvgg import get_RepVGG_func_by_name
from onnxsim import simplify
import argparse


def main(arch, model_path, output_path, input_shape=(224, 224), batch_size=1):
    repvgg_build_func = get_RepVGG_func_by_name(arch)
    model = repvgg_build_func(deploy=True)
    model.load_state_dict(torch.load(model_path))
    dummy_input = torch.autograd.Variable(torch.randn(batch_size, 3, input_shape[0], input_shape[1]))
    torch.onnx.export(model, dummy_input, output_path, verbose=True, keep_initializers_as_inputs=True, opset_version=12)
    onnx_model = onnx.load(output_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')
    parser.add_argument('--weights_file', type=str, default='./RepVGG-A0-deploy.pth', help='weights file path')
    parser.add_argument('--output_file', type=str, default='./RepVGG-A0-simple.onnx', help='onnx file path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[224, 224], help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    main(opt.arch, opt.weights_file, opt.output_file, input_shape=opt.img_size, batch_size=opt.batch_size)

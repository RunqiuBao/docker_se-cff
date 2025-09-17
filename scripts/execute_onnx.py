import time
import torch
import onnxruntime
import numpy
import cv2
import os

import onnx
from onnx import helper


def fix_squeeze_axes(model_path, out_path):
    model = onnx.load(model_path)
    new_nodes = []
    for n in model.graph.node:
        if n.op_type == "Squeeze":
            # check if has old 'axes' attribute
            axes_attrs = [a for a in n.attribute if a.name == "axes"]
            if axes_attrs:
                axes = list(axes_attrs[0].ints)
                # remove old attribute
                n.attribute.remove(axes_attrs[0])
                # add Constant node for axes
                const_name = n.name + "_axes"
                const_node = helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[const_name],
                    value=onnx.helper.make_tensor(
                        name=const_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(axes)],
                        vals=axes,
                    ),
                )
                new_nodes.append(const_node)
                # add input to Squeeze
                n.input.append(const_name)
        new_nodes.append(n)

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    onnx.save(model, out_path)


def execute_onnx_model(
    inputs: dict,
    onnx_model_path: str,
) -> numpy.ndarray:
    """
    This function executes an ONNX model with the given inputs and model path.
    """
    providers = ["CPUExecutionProvider"]
    if "stereo_detection_head" in onnx_model_path:
        import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()
    modelOrtSession = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

    outputs = modelOrtSession.run(None, inputs)[0]  # warm up

    starttime = time.time()
    outputs = modelOrtSession.run(None, inputs)[0]
    print("ONNX model {} inference time: {} sec.".format(os.path.basename(onnx_model_path), time.time() - starttime))

    return outputs


def main():
    model_root_path = "/root/code/docker_pytorch_trainnn/experiments/unitree/"
    # fix_squeeze_axes(os.path.join(model_root_path, "stereo_detection_head.onnx"), os.path.join(model_root_path, "stereo_detection_head_fixed.onnx"))
    # fix_squeeze_axes(os.path.join(model_root_path, "local_tracking_head.onnx"), os.path.join(model_root_path, "local_tracking_head_fixed.onnx"))
    onnx_model_paths = {
        "concentration_net": os.path.join(model_root_path, "concentration_net.onnx"),
        "disp_head": os.path.join(model_root_path, "disp_head.onnx"),
        "objdet_head": os.path.join(model_root_path, "objdet_head.onnx"),
        "stereo_detection_head": os.path.join(model_root_path, "stereo_detection_head.onnx"),
        "local_tracking_head": os.path.join(model_root_path, "local_tracking_head.onnx"),
    }
    onnx_inputs = torch.load("onnx_inputs.pth")
    models_inputs = {
        "concentration_net": onnx_inputs["concentration_net"],
        "disp_head": onnx_inputs["disp_head"],
        "objdet_head": onnx_inputs["objdet_head"],
        "stereo_detection_head": onnx_inputs["stereo_detection_head"],
        "local_tracking_head": onnx_inputs["local_tracking_head"],
    }

    for model_name, model_path in onnx_model_paths.items():
        print("==================================== start to execute onnx model: {} ====================================".format(model_name))
        outputs = execute_onnx_model(models_inputs[model_name], model_path)


if __name__ == "__main__":
    main()

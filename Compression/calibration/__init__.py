"""Calibration helpers for quantisation stages.

INT8 static quantisation needs representative inputs — we feed them
through the model to learn activation-range scale factors. One reader
here handles both the YOLO (letterbox + [0,1]) and RF-DETR
(stretch + ImageNet-norm) recipes so the quantiser sees the same
statistics the live inferencer does.
"""

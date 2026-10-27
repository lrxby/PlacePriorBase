# ---------------------------------------------------------------
# DOTA-v1.0 evaluation config
# Same model and weights as placepriorp2r2_dota1.py, but the test
# loader/evaluator are switched to the trainval split for computing mAP.
_base_ = ['./placepriorp2r2_dota1.py']

test_dataloader = _base_.val_dataloader
test_evaluator = dict(type='DOTAMetric', metric='mAP', _delete_=True)

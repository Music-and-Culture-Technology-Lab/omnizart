
import tqdm
import tensorflow as tf

from omnizart.models.u_net import semantic_segmentation
from omnizart.music.dataset import get_dataset
from omnizart.music.labels import LabelType
from omnizart.callbacks import EarlyStopping, ModelCheckpoint
from omnizart.music.losses import focal_loss, smooth_loss


PROGRESS_BAR_FORMAT = "{desc} - {percentage:3.0f}% |{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]"


def format_num(num, digit=4):
    rounding = f".{digit}g"
    num_str = f"{num:{rounding}}".replace("+0", "+").replace("-0", "-")
    num = str(num)
    return num_str if len(num_str)<len(num) else num


def gen_bar_postfix(result, targets=["loss", "accuracy"], name_transform=["loss", "acc"]):
    info = []
    for target, name in zip(targets, name_transform):
        if target not in result:
            continue
        val = result[target]
        val_str = format_num(val)
        info_str = f"{name}: {val_str}"
        info.append(info_str)
    return ", ".join(info)


def train_steps(model, dataset, steps=None, bar_title=None, validate=False):
    iter_bar = tqdm.tqdm(dataset, total=steps, desc=bar_title, bar_format=PROGRESS_BAR_FORMAT)

    for iters, data in enumerate(iter_bar):
        feat, label = data[:2]  # Assumed the first two elements are feature and label, respectively.
        if validate:
            step_result = model.test_on_batch(feat, label, return_dict=True)
        else:
            step_result = model.train_on_batch(feat, label, return_dict=True)

        if iters == 0:
            # model.metrics_names is only available after the first train_on_batch
            metrics = model.metrics_names
            state = {metric: 0 for metric in metrics}
            state.update({f"{metric}_sum": 0 for metric in metrics})

        for metric in metrics:
            state[f"{metric}_sum"] += step_result[metric]
            state[metric] = state[f"{metric}_sum"] / (iters+1)
        iter_bar.set_postfix_str(gen_bar_postfix(state))

    # Remove metric_sum columns in the state
    state = {metric: state[metric] for metric in metrics}
    return state


def execute_callbacks(callbacks, func_name, **kwargs):
    if callbacks is not None:
        for callback in callbacks:
            getattr(callback, func_name)(**kwargs)


def train_epochs(
    model,
    train_dataset,
    validate_dataset=None,
    epochs=10,
    steps=100,
    val_steps=100,
    callbacks=None,
    **kwargs
):
    history = {"train": [], "validate": []}
    execute_callbacks(callbacks, "_set_model", model=model)
    execute_callbacks(callbacks, "on_train_begin")
    for epoch_idx in range(epochs):
        # Epoch begin
        execute_callbacks(callbacks, "on_epoch_begin")
        if model.stop_training:
            break

        print(f"Epoch: {epoch_idx+1}/{epochs}")

        # Train batch begin
        execute_callbacks(callbacks, "on_train_batch_begin")
        results = train_steps(model, dataset=train_dataset, steps=steps, bar_title="Train   ", **kwargs)

        # Train batch end
        execute_callbacks(callbacks, "on_train_batch_end")
        history["train"].append(results)

        # Test batch begin
        execute_callbacks(callbacks, "on_test_batch_begin")
        val_results = {}
        if validate_dataset is not None:
            val_results = train_steps(
                model, dataset=validate_dataset, steps=val_steps, validate=True, bar_title="Validate", **kwargs
            )

            # Test batch end
            execute_callbacks(callbacks, "on_test_batch_end")
            history["validate"].append(val_results)

        # Epoch end
        execute_callbacks(callbacks, "on_epoch_end", epoch=epoch_idx+1, history=history)

    execute_callbacks(callbacks, "on_train_end")
    return history


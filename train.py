import argparse

from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from data.batch import batch_inputs
from model import music_sincnet
from eval import evaluate
from utils import *
import os


parser = argparse.ArgumentParser(
    description="Sample-level CNN Architectures for Music Auto-tagging."
)
parser.add_argument("--data-dir", type=str, default="./dataset", metavar="PATH")
parser.add_argument(
    "--train-dir",
    type=str,
    default="./log",
    metavar="PATH",
    help="Directory where to write event logs and checkpoints.",
)

parser.add_argument(
    "--alpha", type=int, default=16, metavar="A", help="Amplifying ratio of SE block."
)

parser.add_argument(
    "--batch-size", type=int, default=14, metavar="N", help="Mini-batch size."
)
parser.add_argument(
    "--momentum", type=float, default=0.9, metavar="M", help="Momentum for SGD."
)
parser.add_argument(
    "--lr", type=float, default=0.0000032, metavar="LR", help="Learning rate."
)
parser.add_argument(
    "--lr-decay",
    type=float,
    default=0.2,
    metavar="DC",
    help="Learning rate decay rate.",
)

parser.add_argument(
    "--dropout", type=float, default=0.5, metavar="DO", help="Dropout rate."
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0, metavar="WD", help="Weight decay."
)

parser.add_argument(
    "--initial-stage", type=int, default=0, metavar="N", help="Stage to start training."
)
parser.add_argument(
    "--patience",
    type=int,
    default=2,
    metavar="N",
    help="Stop training stage after #patiences.",
)
parser.add_argument(
    "--num-lr-decays",
    type=int,
    default=5,
    metavar="N",
    help="Number of learning rate decays.",
)

parser.add_argument(
    "--num-audios-per-shard",
    type=int,
    default=100,
    metavar="N",
    help="Number of audios per shard.",
)
parser.add_argument(
    "--num-segments-per-audio",
    type=int,
    default=10,
    metavar="N",
    help="Number of segments per audio.",
)
parser.add_argument(
    "--num-read-threads",
    type=int,
    default=8,
    metavar="N",
    help="Number of TFRecord readers.",
)
parser.add_argument(
    "--sr",
    type=int,
    default=22050,
    metavar="N",
    help="Sampling rate to set maximum frequency range of music-sinc filterbank",
)
parser.add_argument(
    "--filter_size",
    type=int,
    default=2501,
    metavar="N",
    help="Filter size of the MusicSincLayer",
)
parser.add_argument("--gpu", type=str, default="1")
args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    for stage in range(args.initial_stage, args.num_lr_decays):
        stage_train_dir = make_path(args.train_dir, stage)
        previous_stage_train_dirs = [
            make_path(args.train_dir, stage) for stage in range(0, stage)
        ]
        next_stage_train_dir = make_path(args.train_dir, stage + 1)

        # Pass if there is a training directory of the next stage.
        if os.path.isdir(next_stage_train_dir):
            continue

        # Setup the initial learning rate for the stage.
        lr = args.lr * (args.lr_decay ** stage)
        # Create a directory for the stage.
        os.makedirs(stage_train_dir, exist_ok=True)

        # Find the best checkpoint in the current stage.
        ckpt_path, ckpt_epoch, ckpt_val_loss = find_best_checkpoint(stage_train_dir)
        # If there is no checkpoint in the current stage, then find it in the previous stage.
        if ckpt_path is None:
            ckpt_path, ckpt_epoch, ckpt_val_loss = find_best_checkpoint(
                *previous_stage_train_dirs[-1:]
            )

        print(
            "\n=> Start training stage {}: lr={}, train_dir={}".format(
                stage, lr, stage_train_dir
            )
        )
        if ckpt_path:
            print(
                "=> Found a trained model: epoch={}, val_loss={}, path={}".format(
                    ckpt_epoch, ckpt_val_loss, ckpt_path
                )
            )
        else:
            print("=> No trained model found.")

        #       del model
        train(lr, stage_train_dir, ckpt_path, ckpt_epoch + 1)

    print("\n=> Done.\n")


def train(initial_lr, stage_train_dir, ckpt_path=None, initial_epoch=0):
    x_train, y_train = batch_inputs(
        file_pattern=make_path(
            args.data_dir, "tfrecord", "train-????-of-????.tfrecord"
        ),
        batch_size=args.batch_size,
        is_training=True,
        num_read_threads=args.num_read_threads,
        examples_per_shard=args.num_audios_per_shard * args.num_segments_per_audio,
        shard_queue_name="train_filename_queue",
        example_queue_name="train_input_queue",
    )

    x_val, y_val = batch_inputs(
        file_pattern=make_path(args.data_dir, "tfrecord", "val-????-of-????.tfrecord"),
        batch_size=args.batch_size,
        is_training=False,
        num_read_threads=1,
        examples_per_shard=args.num_audios_per_shard * args.num_segments_per_audio,
        shard_queue_name="val_filename_queue",
        example_queue_name="val_input_queue",
    )

    out = music_sincnet(
        x_train,
        filter_size=args.filter_size,
        filter_num=256,
        sr=args.sr,
        amplifying_ratio=args.alpha,
        drop_rate=args.dropout,
        weight_decay=args.weight_decay,
    )

    model = TFRecordModel(inputs=x_train, val_inputs=x_val, outputs=out)
    if ckpt_path:
        print('=> Load weights from "{}".'.format(ckpt_path))
        model.load_weights(ckpt_path)

    model.summary()
    optimizer = SGD(lr=initial_lr, momentum=args.momentum, decay=1e-6, nesterov=True)
    model.compile_tfrecord(
        optimizer=optimizer, loss="binary_crossentropy", y=y_train, y_val=y_val
    )

    tensor_board = TensorBoard(log_dir=stage_train_dir)
    early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience)
    checkpointer = ModelCheckpoint(
        make_path(stage_train_dir, "ckpt-e{epoch:03d}-l{val_loss:.4f}.hdf5"),
        monitor="val_loss",
        save_best_only=True,
    )
    csv_logger = CSVLogger(make_path(stage_train_dir, "training.csv"), append=True)
    num_train, num_val, num_test = 15250, 1529, 4332
    train_steps = calculate_steps(
        num_train, args.num_segments_per_audio, args.batch_size
    )
    val_steps = calculate_steps(num_val, args.num_segments_per_audio, args.batch_size)
    model.fit_tfrecord(
        epochs=100,
        initial_epoch=initial_epoch,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[tensor_board, early_stopping, checkpointer, csv_logger],
    )
    best_ckpt_path, *_ = find_best_checkpoint(stage_train_dir)
    print(
        '=> The end of the stage. Start evaluation on test set using checkpoint "{}".'.format(
            best_ckpt_path
        )
    )
    roc_auc = evaluate(model, args.data_dir, num_test, args.num_audios_per_shard)
    text = open("./log.txt", "a")
    data = "{} : {}".format(best_ckpt_path, roc_auc)
    text.write(data)

    return model


if __name__ == "__main__":
    main()

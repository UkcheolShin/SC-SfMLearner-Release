# for kitti raw dataset
DATASET=/SSD1/Dataset/KITTI/Raw_data/
TRAIN_SET=/SSD1/Dataset_processed/KITTI_256/
STATIC_FILES=data/static_frames.txt
python data/prepare_train_data.py $DATASET --dataset-format 'kitti_raw' --dump-root $TRAIN_SET --width 832 --height 256 --num-threads 16 --static-frames $STATIC_FILES --with-depth  --with-pose

TRAIN_SET=/SSD1/Dataset_processed/KITTI_128/
python data/prepare_train_data.py $DATASET --dataset-format 'kitti_raw' --dump-root $TRAIN_SET --width 416 --height 128 --num-threads 16 --static-frames $STATIC_FILES --with-depth  --with-pose

# # for cityscapes dataset
# DATASET=/media/bjw/Disk/Dataset/cityscapes/
# TRAIN_SET=/media/bjw/Disk/Dataset/cs_256/
# python data/prepare_train_data.py $DATASET --dataset-format 'cityscapes' --dump-root $TRAIN_SET --width 832 --height 342 --num-threads 4

# # for kitti odometry dataset
DATASET=/SSD1/Dataset/KITTI/Odometry/
TRAIN_SET=/SSD1/Dataset_processed/KITTI_VO_256/
python data/prepare_train_data.py $DATASET --dataset-format 'kitti_odom' --dump-root $TRAIN_SET --width 832 --height 256 --num-threads 4

TRAIN_SET=/SSD1/Dataset_processed/KITTI_VO_128/
python data/prepare_train_data.py $DATASET --dataset-format 'kitti_odom' --dump-root $TRAIN_SET --width 416 --height 128 --num-threads 4

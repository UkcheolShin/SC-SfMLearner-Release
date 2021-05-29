DATASET_DIR=/SSD1/Dataset/KITTI/Odometry/dataset/sequences/
OUTPUT_DIR=vo_results/

POSE_NET=checkpoints/resnet18_depth_128/06-18-16\:27/exp_pose_model_best.pth.tar

python test_vo.py \
--img-height 128 --img-width 416 \
--sequence 09 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python test_vo.py \
--img-height 128 --img-width 416 \
--sequence 10 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python ./kitti_eval/eval_odom.py --result=$OUTPUT_DIR --align='7dof'

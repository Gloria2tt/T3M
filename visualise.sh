python -W ignore scripts/diversity.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file ./config/body_pixel.json \
--face_model_path ./experiments/2022-10-15-smplx_S2G-face-3d/ckpt-99.pth \
--body_model_path /root/autodl-tmp/pengwenshuo/TalkSHOW/new_exp_batch256_epoch100/lr5e-5/2023-11-12-smplx_S2G-body-pixel2/ckpt-99.pth \
--infer
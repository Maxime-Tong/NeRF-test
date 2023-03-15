dataset="lego"

# python eval_information.py --config configs/$dataset.txt --render_only --ft_path logs/pre_models/$dataset.tar
python render_visibility.py --config configs/$dataset.txt --render_only --expname lego_render_rgb --ft_path logs/pre_models/$dataset.tar --i_poses 0
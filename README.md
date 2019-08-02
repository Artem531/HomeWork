# You need run main.py from pixor_code

### HomeWork
Please, check this functions:
1) get_learning_data in kitti_datagen_exp..
2) filter_annos in kitti_datagen_exp..
3) get_anno_without_file in kitti_datagen_exp..


I change get_learning_data in kitti_datagen_exp...
Now it receive:
1) param index: index of cloud and anno which you want to aug
2) param number_of_aug: number of augment data
3) param return_clouds: key "return_clouds" if True return optional list of clouds for visualisation
return: list_clouds (if return_clouds == True), list_grid, list_output_class, list_output_reg, list_annos



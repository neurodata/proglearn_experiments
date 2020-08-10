icon_data_resized:
Images resized to 224x224 numpy arrays.
<class_name>.p is an (m, 224, 224, 3) shape array, 
where m is the number of <class_name> examples.

icon_data_encoded:
The resized images encoded by ResNet50's second-to-last layer, as 1000D vectors.
<class_name>.p is an (m, 1000) shape array, 
where m is the number of <class_name> examples.
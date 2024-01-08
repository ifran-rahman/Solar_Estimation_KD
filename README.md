## ENHANCING INTRA-HOUR SOLAR IRRADIANCE ESTIMATION THROUGH KNOWLEDGE DISTILLATION AND INFRARED SKY IMAGES

## Code Organization
All codes are written in python.

* `train_teacher.ipynb` Trains the teacher model. 
* `train_student.ipynb` Trains the student model. 
* `kd.ipynb` Trains the student model with the help of the pre-trained teacher model utilizing 
novel knowledge distillation method. 

### .scripts/
* `saveResults.py` Calculates loss, Creates results folder for particular training based on hyperparameters and time, Saves the results and diagrams.

* `imagetoTensor.py` converts image to pytorch tensor for testing purposes.

* `saveThermalImage.py` converts grayscale infrared image to thermal image using colormap. Used for the menuscript.

### .results/
The output best_models, results and diagrams are stored here.





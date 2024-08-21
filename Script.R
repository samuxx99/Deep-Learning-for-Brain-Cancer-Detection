# Load required libraries
library(keras)
library(tensorflow)
library(reticulate)
library(dplyr)
library(fs)
library(imager)
library(caret)
library(grid)
library(gridExtra)

# Define directories for data
data_dir <- "/home/samuele/Scrivania/Deep Learning /deep learning with R /Brain cancer detection exercise /archive"
test_dir <- file.path(data_dir, "Br35H-Mask-RCNN/TEST")
train_dir <- file.path(data_dir, "Br35H-Mask-RCNN/TRAIN")
val_dir <- file.path(data_dir, "Br35H-Mask-RCNN/VAL")

source_base_dir <- "/home/samuele/Scrivania/Deep Learning /deep learning with R /Brain cancer detection exercise /data" 
dest_base_dir <- "/home/samuele/Scrivania/Deep Learning /deep learning with R /Brain cancer detection exercise /dest"  

yes_dir <- file.path(source_base_dir, "yes")
no_dir <- file.path(source_base_dir, "no")

# Destination directories for train, test, and validation datasets
dest_train_yes <- file.path(dest_base_dir, "train", "yes")
dest_train_no <- file.path(dest_base_dir, "train", "no")
dest_test_yes <- file.path(dest_base_dir, "test", "yes")
dest_test_no <- file.path(dest_base_dir, "test", "no")

# Create destination directories if they do not exist
dir_create(dest_train_yes)
dir_create(dest_train_no)
dir_create(dest_test_yes)
dir_create(dest_test_no)
dir_create(dest_val_yes)
dir_create(dest_val_no)

split_and_copy_exact <- function(src_dir, train_dir, val_dir, test_dir, train_size, val_size, test_size) {
  # List all image files in the source directory
  files <- dir_ls(src_dir, regexp = "\\.(jpg|jpeg|png|bmp)$")
  
  # Check that the total number of files matches the sum of train, val, and test sizes
  if (length(files) != (train_size + val_size + test_size)) {
    stop("The total number of files does not match the sum of the train, val, and test sets.")
  }
  
  # Manually split the files into train, validation, and test sets
  set.seed(123)  # For reproducibility
  train_files <- files[1:train_size]
  val_files <- files[(train_size + 1):(train_size + val_size)]
  test_files <- files[(train_size + val_size + 1):(train_size + val_size + test_size)]
  
  # Copy the files to the corresponding destination directories
  file_copy(train_files, train_dir, overwrite = TRUE)
  file_copy(val_files, val_dir, overwrite = TRUE)
  file_copy(test_files, test_dir, overwrite = TRUE)
}

# Number of images per class
train_size <- 1050
val_size <- 225
test_size <- 225

# Run the function for the 'yes' class
split_and_copy_exact(yes_dir, dest_train_yes, dest_val_yes, dest_test_yes, train_size, val_size, test_size)

# Run the function for the 'no' class
split_and_copy_exact(no_dir, dest_train_no, dest_val_no, test_size, train_size, val_size)


# Load and inspect a sample image
img <- load.image("/home/samuele/Scrivania/Deep Learning /deep learning with R /Brain cancer detection exercise /archive/Br35H-Mask-RCNN/TEST/y702.jpg")
dim(img)


# Retrieve a batch of images and labels from the test generator
batch <- generator_next(test_generator)
images <- batch[[1]]
labels <- batch[[2]]

# Check the dimensions of the objects in the batch
n <- dim(images)[1]
str(images)
dim(images)

images_array <- images[[1]]
dim(images_array)

# Set the number of rows and columns for the image grid
nrows <- 4
ncols <- 4

# List to store the grobs (graphics) of the images
grobs <- list()

# Labels for the images
labels_vector <- labels[[1]]

# Create grobs for each image and its label
for (i in 1:(nrows * ncols)) {
  if (i <= dim(images_array)[1]) {  # Check if there are enough images
    img <- as.raster(images_array[i, , , ])
    label <- ifelse(labels_vector[i] == 1, "Tumor", "No Tumor")
    
    grobs[[i]] <- arrangeGrob(
      rasterGrob(img),
      top = textGrob(label, gp = gpar(fontsize = 10))
    )
  }
}

# Create and display the grid with images and their respective labels
grid_arrange_shared_legend <- grid.arrange(grobs = grobs, nrow = nrows, ncol = ncols)
grid.newpage()
grid.draw(grid_arrange_shared_legend)

# Display a single image with the label "Tumor"
img_2 <- as.raster(images_array[1, , , ])
label_2 <- ifelse(labels_vector[1] == 1, "Tumor", "No Tumor")

g_2 <- arrangeGrob(
  rasterGrob(img_2),
  top = textGrob(label_2, gp = gpar(fontsize = 10))
)

grid.newpage()
grid.draw(g_2)

# Display a single image with the label "No Tumor"
img_3 <- as.raster(images_array[4, , , ])
label_3 <- ifelse(labels_vector[4] == 1, "Tumor", "No Tumor")

g_3 <- arrangeGrob(
  rasterGrob(img_3),
  top = textGrob(label_3, gp = gpar(fontsize = 10))
)

grid.newpage()
grid.draw(g_3)

## Image manipulation and data augmentation

# Define target image dimensions
target_height <- 32
target_width <- 32

# Define data augmentation for training data
train_datagen <- image_data_generator(
  rescale = 1/255, 
  horizontal_flip = TRUE, 
  rotation_range = 20, 
  shear_range = 0.2, 
  zoom_range = 0.2
)

datagen <- image_data_generator(rescale = 1/255)

# Create data generators for training, testing, and validation
train_generator <- flow_images_from_directory(
  directory = "/home/samuele/Scrivania/Deep Learning /deep learning with R /Brain cancer detection exercise /dest/train", 
  target_size = c(target_height, target_width), 
  batch_size = 32, 
  class_mode = "binary", 
  generator = datagen
)

test_generator <- flow_images_from_directory(
  directory = "/home/samuele/Scrivania/Deep Learning /deep learning with R /Brain cancer detection exercise /dest/test", 
  target_size = c(target_height, target_width), 
  batch_size = 32, 
  class_mode = "binary", 
  generator = datagen
)

val_generator <- flow_images_from_directory(
  directory = "/home/samuele/Scrivania/Deep Learning /deep learning with R /Brain cancer detection exercise /dest/val", 
  target_size = c(target_height, target_width), 
  batch_size = 32, 
  class_mode = "binary", 
  generator = datagen
)

## Model 1: Simple CNN

# Define a simple CNN model
input <- layer_input(shape = c(target_height, target_width, 3))
output <- input %>% 
  layer_conv_2d(32, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(128, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_flatten() %>% 
  layer_dense(128, activation = "relu") %>% 
  layer_dense(1, activation = "sigmoid")

model <- keras_model(input, output)

# Compile the model
model %>% compile(
  optimizer = "rmsprop", 
  loss = "binary_crossentropy", 
  metrics = "accuracy"
)

# Train the model
history <- model %>% fit(
  train_generator, 
  epochs = 5, 
  batch_size = 32, 
  validation_data = test_generator
)

# Print model summary and plot training history
model %>% summary()
plot(history)

## Model 2: Transfer Learning with VGG16

# Load pre-trained VGG16 model (excluding top layers)
conv_base <- application_vgg16(
  weights = "imagenet", 
  include_top = FALSE, 
  input_shape = c(target_height, target_width, 3)
)

# Define the model with VGG16 base and custom top layers
input <- layer_input(shape = c(target_height, target_width, 3))
output <- input %>% 
  conv_base() %>% 
  layer_flatten() %>% 
  layer_dense(128, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(input, output)

# Compile the model
model %>% compile(
  optimizer = "adam", 
  loss = "binary_crossentropy", 
  metrics = "accuracy"
)

# Set up callbacks
callbacks <- list(
  callback_early_stopping(monitor = "val_loss", patience = 3),
  callback_model_checkpoint(filepath = "best_model.h5", save_best_only = TRUE)
)

# Train the model
history <- model %>% fit(
  train_generator, 
  epochs = 10, 
  batch_size = 32, 
  validation_data = test_generator, 
  callbacks = callbacks
)

# Plot training history
plot(history)

# Evaluate the model
result <- model %>% evaluate(test_generator)
result

## Model 3: Data Augmentation and VGG16

# Recreate data generators with augmentation
train_generator <- flow_images_from_directory(
  directory = "/home/samuele/Scrivania/Deep Learning /deep learning with R /Brain cancer detection exercise /dest/train", 
  target_size = c(target_height, target_width), 
  batch_size = 32, 
  class_mode = "binary", 
  generator = train_datagen
)

test_generator <- flow_images_from_directory(
  directory = "/home/samuele/Scrivania/Deep Learning /deep learning with R /Brain cancer detection exercise /dest/test", 
  target_size = c(target_height, target_width), 
  batch_size = 32, 
  class_mode = "binary", 
  generator = train_datagen
)

# Define the model with VGG16 base and augmented data
input <- layer_input(shape = c(target_height, target_width, 3))
output <- input %>% 
  conv_base() %>% 
  layer_flatten() %>% 
  layer_dense(128, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(input, output)

# Compile the model
model %>% compile(
  optimizer = "adam", 
  loss = "binary_crossentropy", 
  metrics = "accuracy"
)

# Set up callbacks
callbacks <- list(
  callback_early_stopping(monitor = "val_loss", patience = 3),
  callback_model_checkpoint(filepath = "best_model.h5", save_best_only = TRUE)
)

# Train the model
history <- model %>% fit(
  train_generator, 
  epochs = 10, 
  batch_size = 32, 
  validation_data = test_generator, 
  callbacks = callbacks
)

# Plot training history
plot(history)

# Evaluate the model
result <- model %>% evaluate(test_generator)
result

## Model 4: Limited Data Augmentation

# Adjust data augmentation parameters
train_datagen <- image_data_generator(
  rescale = 1/255, 
  horizontal_flip = FALSE, 
  rotation_range = 5, 
  shear_range = 0.0, 
  zoom_range = 0.0
)

# Create new data generators
train_generator <- flow_images_from_directory(
  directory = "/home/samuele/Scrivania/Deep Learning /deep learning with R /Brain cancer detection exercise /dest/train", 
  target_size = c(target_height, target_width), 
  batch_size = 32, 
  class_mode = "binary", 
  generator = train_datagen
)

# Define the model with VGG16 base and limited data augmentation
input <- layer_input(shape = c(target_height, target_width, 3))
output <- input %>% 
  conv_base() %>% 
  layer_flatten() %>% 
  layer_dense(128, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(input, output)

# Compile the model
model %>% compile(
  optimizer = "adam", 
  loss = "binary_crossentropy", 
  metrics = "accuracy"
)

# Set up callbacks
callbacks <- list(
  callback_early_stopping(monitor = "val_loss", patience = 3),
  callback_model_checkpoint(filepath = "best_model.h5", save_best_only = TRUE)
)

# Train the model
history <- model %>% fit(
  train_generator, 
  epochs = 10, 
  batch_size = 32, 
  validation_data = test_generator, 
  callbacks = callbacks
)

# Plot training history
plot(history)

# Evaluate the model
result <- model %>% evaluate(test_generator)
result

## Model 5: CNN with four convolutional layers
input_5 <- layer_input(shape = c(target_height, target_width, 3))
output_5 <- input_5 %>% 
  layer_conv_2d(32, kernel_size = c(5,5), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_conv_2d(64, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%  
  layer_conv_2d(128, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_conv_2d(256, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_flatten() %>% 
  layer_dropout(.5) %>% 
  layer_dense(256, activation = "relu") %>% 
  layer_dense(1, activation = "sigmoid")

model_5 <- keras_model(inputs = input_5, outputs = output_5)

model_5 %>% compile(optimizer = "adam", 
                    loss = "binary_crossentropy", 
                    metrics = "accuracy")

history_5 <- model_5 %>% fit(train_generator, 
                             validation_data = val_generator,
                             epochs = 35, 
                             callbacks = list(callback_model_checkpoint(filepath = "best_model.h5", save_best_only = TRUE)), 
                             batch_size = 32)

plot(history_5)
result_5 <- evaluate(model_5, test_generator)
result_5

# Debugging
input <- layer_input(shape = c(target_height, target_width, 3))
output <- input %>%
  layer_conv_2d(32, kernel_size = c(5,5), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_conv_2d(64, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%  
  layer_conv_2d(128, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_conv_2d(256, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_flatten() %>%
  layer_lambda(function(x) { print(dim(x)); x }) %>%  # Verifica dimensioni
  layer_dropout(.5) %>% 
  layer_dense(1024, activation = "relu") %>%  # Adatta il numero di neuroni
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs = input, outputs = output)

model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

## Model 6: Retrain with Validation Partition
input <- layer_input(shape = c(target_height, target_width, 3))
output <- input %>% 
  layer_conv_2d(32, kernel_size = c(5,5), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_conv_2d(64, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%  
  layer_conv_2d(128, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_conv_2d(256, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_flatten() %>% 
  layer_dropout(.5) %>% 
  layer_dense(256, activation = "relu") %>% 
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs = input, outputs = output)

model %>% compile(optimizer = "adam", 
                  loss = "binary_crossentropy", 
                  metrics = "accuracy")

callbacks <- list(
  callback_model_checkpoint(filepath = "best_model.h5", save_best_only = TRUE))

history <- model %>% fit(train_generator, 
                         validation_data = val_generator,
                         epochs = 25, 
                         callbacks = callbacks, 
                         batch_size = 32)

plot(history)
result <- evaluate(model, test_generator)
result

# Model 7: Trying More Epochs and Batch Size = 16
input <- layer_input(shape = c(target_height, target_width, 3))
output <- input %>% 
  layer_conv_2d(32, kernel_size = c(5,5), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_conv_2d(64, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%  
  layer_conv_2d(128, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_conv_2d(256, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(.2) %>%
  layer_flatten() %>% 
  layer_dropout(.5) %>% 
  layer_dense(256, activation = "relu") %>% 
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs = input, outputs = output)

model %>% compile(optimizer = "adam", 
                  loss = "binary_crossentropy", 
                  metrics = "accuracy")

callbacks <- list(
  callback_model_checkpoint(filepath = "best_model.h5", save_best_only = TRUE))

history_1 <- model %>% fit(train_generator, 
                           validation_data = val_generator,
                           epochs = 35, 
                           callbacks = callbacks, 
                           batch_size = 16)

plot(history_1)
result <- evaluate(model, test_generator)
result

# Plot the model architecture
grViz("
digraph model {
  graph [layout=dot, fontsize=20]
  node [shape=box, fontsize=20]
  edge [fontsize=20]
  
  input [label='Input Layer']
  conv1 [label='Conv2D (32 filters)']
  pool1 [label='MaxPooling2D']
  dropout1 [label='Dropout']
  conv2 [label='Conv2D (64 filters)']
  pool2 [label='MaxPooling2D']
  dropout2 [label='Dropout']
  conv3 [label='Conv2D (128 filters)']
  pool3 [label='MaxPooling2D']
  dropout3 [label='Dropout']
  conv4 [label='Conv2D (256 filters)']
  pool4 [label='MaxPooling2D']
  dropout4 [label='Dropout']
  flatten [label='Flatten']
  dense1 [label='Dense (256 units)']
  output [label='Output (sigmoid)']

  input -> conv1 -> pool1 -> dropout1 -> conv2 -> pool2 -> dropout2 -> conv3 -> pool3 -> dropout3 -> conv4 -> pool4 -> dropout4 -> flatten -> dense1 -> output
}
")

# BEST MODEL
best_model <- load_model_hdf5("best_model.h5")
result_best <- evaluate(best_model, test_generator)  
result_best

history_data <- history_1$metrics

# Find the epoch with the best validation accuracy
best_epoch <- which.max(history_1$metrics$val_accuracy)
best_accuracy <- history_1$metrics$val_accuracy[best_epoch]
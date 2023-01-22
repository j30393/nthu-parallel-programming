import tensorflow as tf
import horovod.tensorflow as hvd

# TODO[1]: initialize Horovod.
hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
# Since there are no GPU on apollo, this code will not be executed
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# if gpus:
#     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Load Mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

# Create Dataset
dataset = tf.data.Dataset.from_tensor_slices((
    tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
    tf.cast(mnist_labels, tf.int64)))
dataset = dataset.repeat().shuffle(10000).batch(128)

# Build Model
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Create Loss
loss = tf.losses.SparseCategoricalCrossentropy()

# TODO[2]: adjust learning rate based on number of process.
# default learning rate is 0.001
num_processes = hvd.size()
if num_processes > 1:
    lr = 0.001 * num_processes
else:
    lr = 0.001

# Create optimizer
opt = tf.optimizers.Adam(lr)

# checkpoint_dir = './checkpoints'
# checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)


@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        # TODO[3]: broadcast mnist_model.variables and opt.variables() to all other processes.
        mnist_model_vars = mnist_model.variables
        opt_vars = opt.variables()

        hvd.broadcast_variables(mnist_model_vars, root_rank=0)
        hvd.broadcast_variables(opt_vars, root_rank=0)

    return loss_value


print("Number of process: ", hvd.size())

# Start Training
for batch, (images, labels) in enumerate(dataset.take(1000 // hvd.size())):
    loss_value = training_step(images, labels, batch == 0)

    if batch % 10 == 0 and hvd.local_rank() == 0:
        print('Step #%d\tLoss: %.6f' % (batch, loss_value))

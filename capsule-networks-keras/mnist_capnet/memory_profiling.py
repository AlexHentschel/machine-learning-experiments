import os
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'

import tensorflow as tf
from tensorflow.python.client import timeline

x = tf.random_normal([1000, 1000], name="FullMatrix")
_y = tf.random_normal([1, 1000], name="SmallMatrix")
y = tf.tile(_y, [1000,1], name="TiledMatrix")
res = tf.matmul(x, y, name="MatmulMatrix")

# Run the graph with full trace option
with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(res, options=run_options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format(show_memory=True)
    with open('timeline.json', 'w') as f:
        f.write(ctf)


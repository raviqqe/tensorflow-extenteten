import numpy
import tensorflow as tf

import data
import log
import nn
import nn.mlmc



# functions

def predict(train_data,
            test_data,
            hyper_params,
            experiment_setting,
            summary_dir):
  data_info = _analyze_data(train_data, test_data)

  with tf.name_scope("inputs"):
    document_type_and_shape = (tf.int32, (None, data_info["document_length"]))
    x_forward = tf.placeholder(*document_type_and_shape, name="x_forward")
    x_backward = tf.placeholder(*document_type_and_shape, name="x_backward")
    y_true = tf.placeholder(tf.int64,
                            (None, data_info["num_of_labels"]),
                            name="y_true")
    dropout_prob = tf.placeholder(tf.float32, (), name="dropout_prob")

  with tf.name_scope("model"):
    output_layer = nn.char2doc(
      x_forward,
      x_backward,
      char_space_size=data_info["char_space_size"],
      char_embedding_size=hyper_params["character_embedding_size"],
      document_embedding_size=hyper_params["document_embedding_size"],
      dropout_prob=dropout_prob,
      hidden_layer_size=hyper_params["hidden_layer_size"],
      output_layer_size=data_info["num_of_labels"]*data_info["num_of_classes"],
    )
    loss, accuracy, predicted_labels = nn.mlmc.classify(output_layer, y_true)
    loss += nn.regularize_with_l2_loss(hyper_params["l2_regularization_scale"])

  with tf.name_scope("training"):
    do_training = tf.train.AdamOptimizer().minimize(loss)
    train_summary = tf.scalar_summary(["train_accuracy", "train_loss"],
                                      tf.pack([accuracy, loss]))

  with tf.name_scope("test"):
    test_summary = tf.scalar_summary(["test_accuracy", "test_loss"],
                                     tf.pack([accuracy , loss]))

  with tf.Session() as session:
    summarizer = tf.train.SummaryWriter(summary_dir, session.graph_def)
    session.run(tf.initialize_all_variables())

    for epoch in range(experiment_setting["num_of_epochs"]):
      log.message("epoch:", epoch)

      # train

      for batch in data.batches(data.shuffle(train_data),
                                experiment_setting["batch_size"]):
        session.run(do_training, {
          x_forward : batch.forward_documents,
          x_backward : batch.backward_documents,
          y_true : batch.labels,
          dropout_prob : hyper_params["dropout_probability"],
        })

      sampled_train_data = data.sample(train_data, test_data.size)

      summarizer.add_summary(session.run(
        train_summary, {
        x_forward : sampled_train_data.forward_documents,
        x_backward : sampled_train_data.backward_documents,
        y_true : sampled_train_data.labels,
        dropout_prob : 0,
      }), epoch)

      # test

      new_test_summary, last_predicted_labels = session.run(
        tf.tuple((test_summary, predicted_labels)), {
        x_forward : test_data.forward_documents,
        x_backward : test_data.backward_documents,
        y_true : test_data.labels,
        dropout_prob : 0,
      })
      summarizer.add_summary(new_test_summary, epoch)

    return last_predicted_labels


def _analyze_data(train_data, test_data):
  assert train_data.documents.ndim == test_data.documents.ndim == 2 \
         and train_data.documents.shape[1] == test_data.documents.shape[1]
  assert train_data.labels.ndim == test_data.labels.ndim == 2 \
         and train_data.labels.shape[1] == test_data.labels.shape[1]

  all_documents = numpy.concatenate((train_data.documents,
                                     test_data.documents))
  all_labels = numpy.concatenate((train_data.labels, test_data.labels))

  return {
    "char_space_size" : all_documents.max() + 1,
    "document_length" : test_data.documents.shape[1],
    "num_of_labels" : test_data.labels.shape[1],
    "num_of_classes" : all_labels.max() + 1,
  }

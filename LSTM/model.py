import numpy as np
from sklearn import linear_model
import math

from datetime import datetime

from keras.datasets import imdb

max_words_in_voc = 80000 

special_tokens = {0: '<pad>', 
                  1: '<start>', 
                  2: '<oov>'}

(x_train, y_train), (x_test, y_test) = imdb.load_data(
                                          path="imdb.npz",
                                          num_words=max_words_in_voc, # maximum number of indexed word, None = all
                                          skip_top=0, # skip n words with the highest occurance count
                                          maxlen=None, # truncate examples longer then N words
                                          seed=113, # random seed
                                          start_char=1, # index to be used for <start> token
                                          oov_char=2, # index to be used for unindexed words
                                          index_from=len(special_tokens)) # add `index_from` to all inidcies for regular words

for text in x_train:
    for idx, ind in enumerate(text):
        if ind == 10:
            text[idx] = 2

for text in x_test:
    for idx, ind in enumerate(text):
        if ind == 10:
            text[idx] = 2

x_dev = x_train[22000:]
y_dev = y_train[22000:]
x_train = x_train[:22000]
y_train = y_train[:22000]

word2ind = imdb.get_word_index()

ind2word = {ind + len(special_tokens): word for word, ind in word2ind.items()} # each index is shifted by 3, as we stated in the load_imdb function
ind2word.update(special_tokens)
voc_size = min(max_words_in_voc, len(ind2word)) # maximum word index in our dataset + 

word_counts = np.zeros(voc_size)
for text in x_train:
    for word_idx in text:
        word_counts[word_idx] += 1
word_counts[0] = 1
word_counts[3] = 1

def inds2text(inds_list):
    return ' '.join(map(ind2word.get, inds_list))

def get_sample_batch(x, batch_size, start=0):
    batch_X = x[start : start + batch_size]
    return batch_X

from keras.preprocessing import sequence
def pad_batch(batch, max_seq_len):
    batch_padded = sequence.pad_sequences(batch, 
                                          maxlen=max_seq_len, # maximum length of the example
                                          padding='post', # from which end to pad short examples
                                          truncating='post') # from which end to truncate long examples
    return batch_padded

def softmax(x, temperature):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))

def sample(dir, count, epoch_num, sess):
    file_name = dir + '/Samples_epoch_' + str(epoch_num) 
    file = open(file_name, 'w')
    
    X = [[x_train[0][1]]]
    new_word = x_train[0][1]
    feed = {network_dict['ph_X'] : X}
    string = inds2text([new_word]) + ' '
    for i in range(15):
        final_state = sess.run([network_dict['logits'], network_dict['lstm_final_state']], feed_dict = feed)
        new_word = np.argmax(res)
        feed = {network_dict['ph_X'] : [[new_word]], network_dict['initial_state'] : final_state}
        string += inds2text([new_word]) + ' '
    file.write(string + '\n' + '\n')
        
    for i in range(count):
        string = ''
        X = [[1]]
        feed = {network_dict['ph_X'] : X}
        for j in range(20):
            output, final_state = sess.run([network_dict['logits'], network_dict['lstm_final_state']], feed_dict = feed)
            probabilities = softmax(output.reshape(voc_size), 0.5)
            rand_num = np.random.uniform(0, 1, 1)
            
            tmp = 0
            res = 0
            for prob in np.sort(probabilities)[::-1]:
                if prob + tmp > rand_num:
                    res = prob
                    break
                tmp += prob
                
            pos = 0
            for elem in probabilities:
                if elem == res:
                    break
                pos += 1
                
            feed = {network_dict['ph_X'] : [[pos]], network_dict['initial_state'] : final_state}
            string += inds2text([pos]) + ' '
            #print(inds2text([pos]))
        file.write(string + '\n' + '\n')


import tensorflow as tf

tf.reset_default_graph()

def create_embedding_layer(hyper_parameters, 
                           network_dict):
    var_embs = tf.get_variable('var_embs', shape=[hyper_parameters['voc_size'], 
                                                                  hyper_parameters['emb_size']],
                                               dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(-1, 1),
                                               trainable=True)

    print(var_embs.name, var_embs.shape)
    
    
    el = tf.nn.embedding_lookup(var_embs, network_dict['ph_X'], name = 'embedding_lookup') # (bs, max_len, emb_size)
    print(el.name, el.shape)
    network_dict['word_embeddings'] =  tf.nn.dropout(el, network_dict['ph_dropout'], name = 'dropout')
    print(network_dict['word_embeddings'].name, network_dict['word_embeddings'].shape)

def create_training_operation(hyper_parameters, network_dict):
    global_step = tf.Variable(0, trainable=False)
    network_dict['global_step'] = global_step

    lr = tf.train.exponential_decay(hyper_parameters['starting_lr'],
                                    global_step = global_step,
                                    decay_steps = 1,
                                    decay_rate = hyper_parameters['decay_rate'])
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    hyper_parameters['lr'] = lr
    grads_vars = optimizer.compute_gradients(network_dict['loss'])
      
    
    non_embedding_grads_and_vars = [(g, v) for (g, v) in grads_vars if 'var_embs' not in v.op.name]
    embedding_grads_and_vars = [(g, v) for (g, v) in grads_vars if 'var_embs' in v.op.name]

    ne_grads, ne_vars = zip(*non_embedding_grads_and_vars)
    ne_grads, _ = tf.clip_by_global_norm(ne_grads, 1)
    non_embedding_grads_and_vars = list(zip(ne_grads, ne_vars))

    clipped_grads_vars = embedding_grads_and_vars + non_embedding_grads_and_vars
    
    
    network_dict['grads_vars'] = grads_vars
    
    network_dict['train_op'] = optimizer.apply_gradients(clipped_grads_vars, global_step = global_step)

def create_placeholders(hyper_parameters,
                        network_dict):
    ph_X = tf.placeholder(shape=(hyper_parameters['bs'], hyper_parameters['max_len']), 
                          dtype=tf.int64, 
                          name="text_input")

    print(ph_X.name, ph_X.shape)
    network_dict['ph_X'] = ph_X
    network_dict['ph_y'] = tf.placeholder(shape=(hyper_parameters['bs'], hyper_parameters['max_len']), 
                                          dtype=tf.int64, 
                                          name="lm_labels")

    print(network_dict['ph_y'].name, network_dict['ph_y'].shape)

    # dynamically retrieve batch size in runtime (required, when we work with batch_size = None)
    network_dict['bs_dynamic'] = tf.shape(ph_X)[0]
    
    # retrieve lengths of the examples during runtime
    non_zero_ids = tf.not_equal(ph_X, 0)
    non_zero_total = tf.reduce_sum(tf.cast(non_zero_ids, tf.int32), axis = 1)
    network_dict['example_lens'] = non_zero_total

    network_dict['ph_dropout'] = tf.placeholder(shape=(), dtype=tf.float32, name = 'dropout_coef')
    print(network_dict['ph_dropout'].name, network_dict['ph_dropout'].shape)


    network_dict['ph_labels'] = tf.placeholder(shape=(hyper_parameters['bs'], 1), 
                                dtype=tf.float32)
    
    network_dict['ph_loss_summary'] = tf.placeholder(shape=None, dtype=tf.float32, name='loss_placeholder')
    
def create_lstm(hyper_parameters, network_dict):
    word_embeddings = network_dict['word_embeddings'] # (batch_size, max_len, embedding_size)

    ph_X = network_dict['ph_X']
    bs = network_dict['bs_dynamic']
    num_units = hyper_parameters['cell_size']
    example_lens = network_dict['example_lens']
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
    network_dict['lstm_cell'] = lstm_cell

    
    
    outputs = []
    
    # actually, this operation creates a placeholder
    tf_zero_state = lstm_cell.zero_state(batch_size = bs, dtype=tf.float32)
    print(tf_zero_state[0].name, tf_zero_state[0].shape)
    print(tf_zero_state[1].name, tf_zero_state[1].shape)

    network_dict['initial_state'] = tf_zero_state
    
    network_outputs, network_state = tf.nn.dynamic_rnn(inputs=word_embeddings, 
                                                       cell=lstm_cell, 
                                                       initial_state=tf_zero_state,
                                                       sequence_length=example_lens)



    print(network_outputs.name, network_outputs.shape)
    print(network_state[0].name, network_state[0].shape)
    print(network_state[1].name, network_state[1].shape)


    network_dict['lstm_outputs'] = network_outputs # bs , max_len, lstm_size
    network_dict['lstm_final_state'] = network_state # LSTMStateTuple (bs, 512)

def create_loss_function(hyper_parameters, network_dict):
    lstm_outputs = network_dict['lstm_outputs'] 
    target_words = network_dict['ph_y'] 
    example_lens = network_dict['example_lens']
    print('\ncreating_loss_function')
    """
    lm_logits = tf.contrib.layers.fully_connected(lstm_outputs, 
                                                  num_outputs=voc_size,
                                                  activation_fn=None) # bs , max_len, voc_size


    network_dict['logits'] = lm_logits
    """
    sequence_mask = tf.sequence_mask(example_lens, maxlen=hyper_parameters['max_len'], dtype=tf.float32)
    #mask = tf.reshape(sequence_mask, [-1])

    W = tf.get_variable(shape=[hyper_parameters['voc_size'], hyper_parameters['cell_size']], 
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        name='weights')
    b = tf.get_variable(shape=[hyper_parameters['voc_size'],], 
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        name='biases')
    print(W.name, W.shape)
    print(b.name, b.shape)

    
    target_reshaped = tf.reshape(target_words, [-1, 1])

    sampled = tf.nn.fixed_unigram_candidate_sampler(
          true_classes=target_reshaped,
          num_true=1,
          num_sampled=hyper_parameters['num_sampled'],
          unique=True,
          range_max=hyper_parameters['voc_size'],
          unigrams=list(word_counts))


    
    #outputs_reshaped = tf.reshape(lstm_outputs, [-1, hyper_parameters['cell_size']])

    
    def my_sampled_softmax(labels, logits):
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.reshape(logits, [-1, hyper_parameters['cell_size']])
        tmp = tf.nn.sampled_softmax_loss(W, b, labels, logits,
                                            hyper_parameters['num_sampled'],
                                            hyper_parameters['voc_size'], 
                                            sampled_values=sampled)
        print(tmp.name, tmp.shape)
        return tmp 
    
    #train_loss = tf.nn.sampled_softmax_loss(W, b, target_reshaped, outputs_reshaped,
    #                                        hyper_parameters['num_sampled'],
    #                                        hyper_parameters['voc_size'], 
    #                                        sampled_values=sampled)

    #network_dict['tl'] = train_loss
    #network_dict['loss'] = tf.reduce_sum(tf.multiply(train_loss, mask)) / tf.reduce_sum(mask)

    

    loss = tf.contrib.seq2seq.sequence_loss(logits=lstm_outputs, 
                                            targets=target_words, 
                                            weights=sequence_mask,
                                            average_across_batch=True, 
                                            average_across_timesteps=True,
                                            softmax_loss_function=my_sampled_softmax,
                                            name='sequence_loss')
    print(loss.name, loss.shape)
    #network_dict['loss'] = tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(example_lens), dtype=tf.float32)
    #network_dict['loss'] = tf.reduce_sum(loss)
    network_dict['loss'] = loss


def train_step(network_dict, input_segements, tf_session, do_backward=True):
    loss = network_dict['loss']
    lstm_final_state = network_dict['lstm_final_state']
    ph_X = network_dict['ph_X']
    ph_y = network_dict['ph_y']
    dropout_coef = network_dict['ph_dropout']
    
    summ = network_dict['summary']
    gs = network_dict['global_step']
    #lr_summary = network_dict['lr_summary']
    
    lstm_init_state = network_dict['initial_state']

    feed_dict = {}
    tmp = 0
    if do_backward:
        train_op = network_dict['train_op']
        run_ops = [gs, lstm_final_state, loss, train_op]
        feed_dict[dropout_coef] = hyper_parameters['dropout_coef']
    else:
        run_ops = [lstm_final_state, loss]
        tmp = -1
        feed_dict[dropout_coef] = 1


    losses = []
    segments_X, segments_y = input_segements
    for i in range(len(segments_X)):
        feed_dict[ph_X] = segments_X[i]
        feed_dict[ph_y] = segments_y[i]
        if i != 0:
            feed_dict[lstm_init_state] = final_state
        result = tf_session.run(run_ops, feed_dict=feed_dict)


        final_state = result[1 + tmp]
        loss = result[2 + tmp]
        #losses.append(loss * segments_X[i].shape[1]) # loss * seg_len
        losses.append(loss)


        
    final_loss = np.sum(losses) / len(segments_X)
    if do_backward:
        result = tf_session.run([gs, summ], {network_dict['ph_loss_summary']:final_loss})
    else:
        result = tf_session.run([summ], {network_dict['ph_loss_summary']:final_loss})
    if do_backward:
        #return result[0], result[1], np.sum(losses) / np.sum([segment.shape[1] for segment in segments_X])
        return result[0], result[1], np.sum(losses) / len(segments_X)
    else:
        #return result[0], np.sum(losses) / np.sum([segment.shape[1] for segment in segments_X])
        return result[0], np.sum(losses) / len(segments_X)



def padded_segments(batch_X, segment_size):
    sent_lens = map(len, batch_X)
    segments_X = []
    segments_y = []
    num_segments = int(np.ceil(max(sent_lens) / segment_size))
    
    tmp = num_segments * 300
    batch_padded_X = sequence.pad_sequences(batch_X, 
                                          maxlen=tmp, # num_segments * segment_size, 
                                          padding='post', 
                                          truncating='post')
    batch_padded_y = np.concatenate([batch_padded_X, 
                                     np.zeros(shape=(batch_padded_X.shape[0], 1), dtype='int')], axis=1)
    batch_padded_y = batch_padded_y[:, 1:]
    for i in range(num_segments):
        segments_X.append(batch_padded_X[:, i * segment_size: (i+1) * segment_size])
        segments_y.append(batch_padded_y[:, i * segment_size: (i+1) * segment_size])
    return segments_X, segments_y

def create_network(hyper_parameters):
    tf.reset_default_graph()
    network_dict = dict()
    create_placeholders(hyper_parameters, network_dict)
    create_embedding_layer(hyper_parameters, network_dict)
    create_lstm(hyper_parameters, network_dict)
    create_loss_function(hyper_parameters, network_dict)
    create_training_operation(hyper_parameters, network_dict)
    create_predictions(network_dict)
    create_predictions_train_op(network_dict)
    return network_dict

def create_predictions(network_dict):
    layer_size = 30
    layer = tf.contrib.layers.fully_connected(network_dict['lstm_final_state'].h, layer_size,
                                                   biases_initializer=tf.random_normal_initializer(mean=0, stddev=math.sqrt(2. / hyper_parameters['cell_size'])),
                                                   activation_fn=tf.nn.relu)
    
    prediction = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None,
                                                   biases_initializer=tf.contrib.layers.xavier_initializer())
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=network_dict['ph_labels'], logits=prediction)
    network_dict['prd_loss'] = tf.reduce_mean(loss)
    network_dict['preds'] = prediction

def create_predictions_train_op(network_dict):
    start_lr = 0.0005
    global_step_pred = tf.Variable(0, trainable=False)
    network_dict['global_step_pred'] = global_step_pred

    lr_pred = tf.train.exponential_decay(start_lr,
                                    global_step = global_step_pred,
                                    decay_steps = 1,
                                    decay_rate = 0.9998)
    hyper_parameters['lr_pred'] = lr_pred
    
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_pred)
    network_dict['pred_op'] = optimizer.minimize(network_dict['prd_loss'], global_step=global_step_pred)

def additional_metrics(hyper_parameters, network_dict, sess, writer_train, writer_dev, gs):
    single_neuron_classifiers = [linear_model.LogisticRegression() for i in range(hyper_parameters['cell_size'])]
    all_neurons_classifier = linear_model.LogisticRegressionCV()
    
    final_states = []
    for j in range(batch_count):
        batch = get_sample_batch(x_train, batch_size, j * batch_size)
        x_seg = padded_segments(batch, segment_len)[0]
        final_state = []
        for i in range(len(x_seg)):
            feed = {network_dict['ph_X'] : x_seg[i]}
            feed[network_dict['ph_dropout']] = 1
            if i != 0:
                feed[network_dict['initial_state']] = final_state
            final_state = sess.run(network_dict['lstm_final_state'], feed)
        c, h = final_state
        for state in c:
            final_states.append(state)

    final_states = np.array(final_states)


    all_neurons_classifier.fit(final_states, y_train)
    best_res = 0
    best_res_idx = 0
    for idx, clf in enumerate(single_neuron_classifiers):
        clf.fit(final_states[:,idx:idx+1], y_train)
        res = np.sum(clf.predict(final_states[:,idx:idx+1]) == y_train)
        if res > best_res:
            best_res = res
            best_res_idx = idx
            
    all_train_res = np.sum(all_neurons_classifier.predict(final_states) == y_train) / len(x_train)
    single_train_res = best_res / len(x_train)
    feed = {ph_all_neurons : all_train_res, ph_single_neuron : single_train_res}
    s = sess.run(network_dict['clf_summary_op'], feed_dict = feed)
    writer_train.add_summary(s, gs)
    
    
    #dev
    final_states = []
    for j in range(dev_batch_count):
        batch = get_sample_batch(x_dev, batch_size, j * batch_size)
        x_seg = padded_segments(batch, segment_len)[0]
        final_state = []
        for i in range(len(x_seg)):
            feed = {network_dict['ph_X'] : x_seg[i]}
            feed[network_dict['ph_dropout']] = 1
            if i != 0:
                feed[network_dict['initial_state']] = final_state
            final_state = sess.run(network_dict['lstm_final_state'], feed)
        c, h = final_state
        for state in c:
            final_states.append(state)

    final_states = np.array(final_states)
    
    all_res = np.sum(all_neurons_classifier.predict(final_states) == y_dev) / len(x_dev)
    single_res = np.sum(single_neuron_classifiers[best_res_idx].predict(final_states[:,best_res_idx:best_res_idx + 1]) == y_dev) / len(x_dev)
    
    feed = {ph_all_neurons : all_res, ph_single_neuron : single_res}
    s = sess.run(network_dict['clf_summary_op'], feed_dict = feed)
    writer_dev.add_summary(s, gs)

def evaluate_dev(network_dict, dev_batch_count, tf_session):
    loss = network_dict['loss']
    lstm_final_state = network_dict['lstm_final_state']
    ph_X = network_dict['ph_X']
    ph_y = network_dict['ph_y']
    dropout_coef = network_dict['ph_dropout']
    
    summ = network_dict['summary']
    gs = network_dict['global_step']
    #lr_summary = network_dict['lr_summary']
    
    lstm_init_state = network_dict['initial_state']

    feed_dict = {}
    run_ops = [lstm_final_state, loss]
    feed_dict[dropout_coef] = 1


    total_loss = []
    result = []
    for batch_num in range(dev_batch_count):
        batch = get_sample_batch(x_dev, batch_size, batch_num * batch_size)
        segments_X, segments_y = padded_segments(batch, segment_len)
        losses = []
        for i in range(len(segments_X)):
            feed_dict[ph_X] = segments_X[i]
            feed_dict[ph_y] = segments_y[i]
            if i != 0:
                feed_dict[lstm_init_state] = final_state
            result = tf_session.run(run_ops, feed_dict=feed_dict)
            final_state = result[0]
            loss = result[1]
            #losses.append(loss * segments_X[i].shape[1]) # loss * seg_len
            losses.append(loss)
        total_loss.append(np.sum(losses) / len(segments_X))
        
    final_loss = np.sum(total_loss) / dev_batch_count
    result = tf_session.run([summ], {network_dict['ph_loss_summary']:final_loss})
    return result[0], final_loss


hyper_parameters = {'bs': 200,
                    'voc_size': voc_size,
                    'cell_size': 1024, 
                    'emb_size': 256, 
                    'max_len': 400,
                    'starting_lr': 0.001,
                    'dropout_coef': 0.5,
                    'decay_rate':0.9999,
                    'num_sampled': 1000}


batch_size = 200
segment_len = 400
batch_count = np.ceil(len(x_train) / batch_size).astype(np.int32)
dev_batch_count = np.ceil(len(x_dev) / batch_size).astype(np.int32)


network_dict = create_network(hyper_parameters)

ph_all_neurons = tf.placeholder(tf.float32)
ph_single_neuron = tf.placeholder(tf.float32)

cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
dir_name = './tensorboard/' + cur_time

writer_train = tf.summary.FileWriter(dir_name + '/train')
writer_train.add_graph(tf.get_default_graph())

writer_dev = tf.summary.FileWriter(dir_name + '/dev')
writer_dev.add_graph(tf.get_default_graph())

_ = tf.summary.scalar('loss', network_dict['ph_loss_summary'])

perplexity = tf.exp(network_dict['ph_loss_summary'])
_ = tf.summary.scalar('perplexity', perplexity)

lr_summ = tf.summary.scalar('learning_rate', hyper_parameters['lr'])

#_ = tf.summary.histogram('Internal_matrix_gradients_before_clipping', network_dict['internal_grads_before'])
#_ = tf.summary.histogram('Logits_gradients_before_clipping', network_dict['logits_grads_before'])

#_ = tf.summary.histogram('Internal_matrix_gradients_after_clipping', network_dict['internal_grads_after'])
#_ = tf.summary.histogram('Logits_gradients_after_clipping', network_dict['logits_grads_after'])

summary = tf.summary.merge_all()


pred_loss_summ = tf.summary.scalar('prediction_loss', network_dict['prd_loss'])

pred_summary = [pred_loss_summ]

pred_summary_op = tf.summary.merge(pred_summary)
network_dict['pred_summ_op'] = pred_summary_op


all_neurons_summ = tf.summary.scalar('all_neurons_classifier_accuracy', ph_all_neurons)
single_neuron_summ = tf.summary.scalar('single_neuron_best_classifier_accuracy', ph_single_neuron)
clf_summary = [all_neurons_summ, single_neuron_summ]

clf_summary_op = tf.summary.merge(clf_summary)

network_dict['clf_summary_op'] = clf_summary_op
network_dict['summary'] = summary
#network_dict['lr_summary'] = lr_summ

file_name = dir_name + '/hyper.txt'
file = open(file_name, 'w')
file.write('Learning rate = ' + str(hyper_parameters['starting_lr']) + '\n')
file.write('LSTM cell size = '+ str(hyper_parameters['cell_size']) + '\n')
file.write('Embedding size = '+ str(hyper_parameters['emb_size']) + '\n')
file.write('Batch size = '+ str(batch_size) + '\n')
file.write('Vocabulary size = '+ str(hyper_parameters['voc_size']) + '\n')


file.close()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
epoch_count = 0
dev_batch = 0




while epoch_count < 50:
    #sample(dir_name, 10, epoch_count, sess)
    for j in range(batch_count):
        batch = get_sample_batch(x_train, batch_size, j * batch_size)
        segments = padded_segments(batch, segment_len)

        global_step, loss_summary, loss_value = train_step(network_dict, segments, sess, do_backward=True)
        print("Loss = %f" % loss_value)
        if j % 10 == 0:
            writer_train.add_summary(loss_summary, global_step)
            if j % 150 == 0:
                ls, dev_loss = evaluate_dev(network_dict, dev_batch_count, sess)
                writer_dev.add_summary(ls, global_step)

            if (epoch_count * batch_count + j) % 1000 == 0:
                additional_metrics(hyper_parameters, network_dict, sess, writer_train, writer_dev, global_step)
    epoch_count += 1



def classifier_train_step(do_backward, network_dict, hyper_parameters, segments, labels):
    prd_loss = network_dict['prd_loss']
    op = network_dict['pred_op']
    state = network_dict['lstm_final_state']
    ph_X = network_dict['ph_X']
    ph_labels = network_dict['ph_labels']
    init_state = network_dict['initial_state']
    summ = network_dict['pred_summ_op']
    global_step = network_dict['global_step_pred']
    
    last_state = network_dict['initial_state']
    seg_count = len(segments)
    for i in range(seg_count - 1):
        if i == 0:
            last_state = sess.run(state, {ph_X:segments[i], network_dict['ph_dropout']:1})
        else:
            last_state = sess.run(state, {ph_X:segments[i], init_state:last_state, network_dict['ph_dropout']:1})
         
        
    feed = {ph_labels:labels, 
            ph_X:segments[seg_count - 1],
            network_dict['ph_dropout']:1}   
    if last_state != network_dict['initial_state']:
        feed[init_state] = last_state
    
    if do_backward:
        gs, summ, loss, _ = sess.run([global_step, summ, prd_loss, op], feed)
    else:
        gs, summ, loss = sess.run([global_step, summ, prd_loss], feed)
        
    #gs, loss, _ = sess.run([global_step, prd_loss, op], feed)
    return gs, summ, loss





batch_size = 40
batch_count = np.ceil(len(x_train) / batch_size).astype(np.int32)
dev_batch_count = np.ceil(len(x_dev) / batch_size).astype(np.int32)
dev_batch = 0

for i in range(100):
    for j in range(batch_count):
        x = get_sample_batch(x_train, batch_size, j * batch_size)
        y = get_sample_batch(y_train, batch_size, j * batch_size)
        y = y.reshape((len(y),1))
        segments = padded_segments(x, 400)[0]
        global_step, summ, loss = classifier_train_step(True, network_dict, hyper_parameters, segments, y)
        if j % 10 == 0: 
            writer_train.add_summary(summ, global_step)
            x = get_sample_batch(x_dev, batch_size, dev_batch * batch_size)
            y = get_sample_batch(y_dev, batch_size, dev_batch * batch_size)
            y = y.reshape((len(y),1))
            segments = padded_segments(x, 400)[0]
            global_step, summ, loss = classifier_train_step(False, network_dict, hyper_parameters, segments, y)
            writer_dev.add_summary(summ, global_step)
        #print(loss)





batch_count = np.ceil(len(x_test) / batch_size).astype(np.int32)
correct = 0
for j in range(batch_count):
    x = get_sample_batch(x_test, batch_size, j * batch_size)
    y = get_sample_batch(y_test, batch_size, j * batch_size)
    y = y.reshape((len(y),1))
    segments = padded_segments(x, 5000)[0]
    pred = sess.run(network_dict['preds'], {network_dict['ph_X']:segments[0]})
    correct += np.sum((pred  >= 0) == y)
    
file_name = dir_name + '/test_accuracy.txt'
file = open(file_name, 'w')
file.write(str(correct / len(x_train)))
file.close()


saver = tf.train.Saver()
save_path = saver.save(sess, dir_name + "/save/model.ckpt")

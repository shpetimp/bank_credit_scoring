def forward_back_prop(rnn, optimizer, criterion, inputs, target, hidden):
    #print('forward back prop')
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    # move model to GPU, if available
    if(train_on_gpu):
        rnn.cuda()
        
#     # Creating new variables for the hidden state, otherwise
#     # we'd backprop through the entire training history
    h = tuple([each.data for each in hidden])

    # zero accumulated gradients
    rnn.zero_grad()
    try:
        print("trying inputs.long.cuda()")
        if(train_on_gpu):
            inputs = inputs.float()
            inputs = inputs.cuda()
            target = target.cuda()
    except RuntimeError:
        raise
#     print(h[0].data)
    print('passed thru inputs.long()')
    # get predicted outputs
    
    
    #print("trying rnn(inputs,h)")
        # get the output from the model
    output, hidden = rnn(inputs, h)        
    
    #output, hidden = rnn(inputs, h)   
        
    
    #output, h = rnn(inputs, h)
    
    # calculate loss
    loss = criterion(output, target)
    
#     optimizer.zero_grad()
    loss.backward()
    # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs / LSTMs
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)

    optimizer.step()
    return loss.item(), h

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)

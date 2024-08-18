# Information Box

During our survey, we found that it is quite challenging to conduct fair comparisons among different PEFT algorithms.

1. Under the same settings, Method A performs poorly in some papers but shows better results in others.

2. Various papers might use different settings for different tasks, involving factors such as the selection of layers for fine-tuning, the size of the trainable parameters, and the distinction between task datasets, among others.

3. We found that some methods could not reproduce the same results as reported in the original papers during our replication attempt, and they also haven't release their codes...

To address these issues, we have decided to make a preliminary attempt. We will test different methods under the same settings. Given that tasks like NLU are highly influenced by random seeds, we have chosen to conduct a unified evaluation on commonsense reasoning tasks. The results we report will satisfy one of the three conditions:

1. The method’s code is open-source, and a download link for the CR task weights is available. We have verified that these weights produce results consistent with the original paper.

2. The method’s code is open-source, but no official weights have been released. We report the results obtained by transferring the official code on CR task.

3. The method is not open-source, but we have successfully reproduced the results consistent with the original paper.

In conclusion, we will not report any results that we are unable to reproduce. We will also provide the weights corresponding to the CR results if they are reported in our paper.
We hope that our attempt can assist researchers and contribute to the advancement of this field.

*Note*: We will create a table here in several weeks. Please wait patiently...


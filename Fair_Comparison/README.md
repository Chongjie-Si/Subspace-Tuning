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

## Results

<table>
    <thead>
        <tr>
            <th>Method</th>
            <th>Params</th>
            <th>BoolQ</th>
            <th>PIQA</th>
            <th>SIQA</th>
            <th>HellaS.</th>
            <th>WinoG.</th>
            <th>ARC-e</th>
            <th>ARC-c</th>
            <th>OBQA</th>
            <th>Avg.</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ChatGPT</td>
            <td>-</td>
            <td>73.1</td>
            <td>85.4</td>
            <td>68.5</td>
            <td>78.5</td>
            <td>66.1</td>
            <td>89.8</td>
            <td>79.9</td>
            <td>74.8</td>
            <td>77.0</td>
        </tr>
        <tr>
            <td colspan="11"><i>Fine-tuning LLaMA-7B</i></td>
        </tr>
        <tr>
            <td>Fully FT</td>
            <td>100%</td>
            <td>69.9</td>
            <td>84.2</td>
            <td>78.9</td>
            <td>92.3</td>
            <td>83.3</td>
            <td>86.6</td>
            <td>72.8</td>
            <td>83.4</td>
            <td>81.4</td>
        </tr>
        <tr>
            <td>Prefix</td>
            <td>0.11%</td>
            <td>64.3</td>
            <td>76.8</td>
            <td>73.9</td>
            <td>42.1</td>
            <td>72.1</td>
            <td>72.9</td>
            <td>54.0</td>
            <td>60.6</td>
            <td>64.6</td>
        </tr>
        <tr>
            <td>Series</td>
            <td>0.99%</td>
            <td>63.0</td>
            <td>79.2</td>
            <td>76.3</td>
            <td>67.9</td>
            <td>75.7</td>
            <td>74.5</td>
            <td>57.1</td>
            <td>72.4</td>
            <td>70.8</td>
        </tr>
        <tr>
            <td>Parallel</td>
            <td>3.54%</td>
            <td>67.9</td>
            <td>76.4</td>
            <td>78.8</td>
            <td>69.8</td>
            <td>78.9</td>
            <td>73.7</td>
            <td>57.3</td>
            <td>75.2</td>
            <td>72.2</td>
        </tr>
        <tr>
            <td>LoRA<sub>r=4</sub></td>
            <td>0.10%</td>
            <td>2.3</td>
            <td>46.1</td>
            <td>18.3</td>
            <td>19.7</td>
            <td>55.2</td>
            <td>65.4</td>
            <td>51.9</td>
            <td>57.0</td>
            <td>39.5</td>
        </tr>
        <tr>
            <td>AdaLoRA<sub>r=4</sub></td>
            <td>+ 0.6k</td>
            <td>66.1</td>
            <td>78.1</td>
            <td>74.3</td>
            <td>34.0</td>
            <td>74.4</td>
            <td>76.7</td>
            <td>57.5</td>
            <td>71.2</td>
            <td>66.5</td>
        </tr>
        <tr>
            <td>FLoRA<sub>r=4</sub></td>
            <td>+ 2.6k</td>
            <td>67.2</td>
            <td>78.0</td>
            <td>72.9</td>
            <td>65.4</td>
            <td>73.8</td>
            <td>73.8</td>
            <td>55.3</td>
            <td>71.8</td>
            <td>69.8</td>
        </tr>
        <tr>
            <td>DoRA<sub>r=4</sub></td>
            <td>+ 877k</td>
            <td>51.3</td>
            <td>42.2</td>
            <td>77.8</td>
            <td>25.4</td>
            <td>78.8</td>
            <td>78.7</td>
            <td>62.5</td>
            <td>78.6</td>
            <td>61.9</td>
        </tr>
        <tr>
            <td>LoRA-Dash<sub>r=4</sub></td>
            <td>+ 1.3k</td>
            <td>65.2</td>
            <td>79.9</td>
            <td>78.3</td>
            <td>82.8</td>
            <td>77.1</td>
            <td>78.6</td>
            <td>65.4</td>
            <td>78.4</td>
            <td>75.7</td>
        </tr>
        <tr>
            <td>LoRA<sub>r=32</sub></td>
            <td>0.83%</td>
            <td>68.9</td>
            <td>80.7</td>
            <td>77.4</td>
            <td>78.1</td>
            <td>78.8</td>
            <td>77.8</td>
            <td>61.3</td>
            <td>74.8</td>
            <td>74.7</td>
        </tr>
        <tr>
            <td>AdaLoRA<sub>r=32</sub></td>
            <td>+ 5.1k</td>
            <td>69.1</td>
            <td>82.2</td>
            <td>77.2</td>
            <td>78.3</td>
            <td>78.2</td>
            <td>79.7</td>
            <td>61.9</td>
            <td>77.2</td>
            <td>75.5</td>
        </tr>
        <tr>
            <td>FLoRA<sub>r=32</sub></td>
            <td>+ 164k</td>
            <td>66.4</td>
            <td>81.3</td>
            <td>77.1</td>
            <td>75.6</td>
            <td>77.1</td>
            <td>77.2</td>
            <td>62.4</td>
            <td>77.6</td>
            <td>74.3</td>
        </tr>
        <tr>
            <td>DoRA<sub>r=32</sub></td>
            <td>+ 877k</td>
            <td>69.7</td>
            <td>83.4</td>
            <td>78.6</td>
            <td>87.2</td>
            <td>81.0</td>
            <td>81.9</td>
            <td>66.2</td>
            <td>79.2</td>
            <td>78.4</td>
        </tr>
        <tr>
            <td>LoRA-Dash<sub>r=32</sub></td>
            <td>+ 1.3k</td>
            <td>69.9</td>
            <td>82.8</td>
            <td>78.6</td>
            <td>84.9</td>
            <td>81.6</td>
            <td>82.3</td>
            <td>66.5</td>
            <td>80.8</td>
            <td>78.4</td>
        </tr>
        <tr>
            <td colspan="11"><i>Fine-tuning LLaMA3-8B</i></td>
        </tr>
        <tr>
            <td>LoRA<sub>r=16</sub></td>
            <td>0.35%</td>
            <td>72.3</td>
            <td>86.7</td>
            <td>79.3</td>
            <td>93.5</td>
            <td>84.8</td>
            <td>87.7</td>
            <td>75.7</td>
            <td>82.8</td>
            <td>82.8</td>
        </tr>
        <tr>
            <td>AdaLoRA<sub>r=16</sub></td>
            <td>+ 2.6k</td>
            <td>90.4</td>
            <td>85.0</td>
            <td>76.7</td>
            <td>79.1</td>
            <td>83.3</td>
            <td>86.4</td>
            <td>75.1</td>
            <td>75.4</td>
            <td>81.4</td>
        </tr>
        <tr>
            <td>FLoRA<sub>r=16</sub></td>
            <td>+ 41k</td>
            <td>90.2</td>
            <td>84.2</td>
            <td>79.9</td>
            <td>79.3</td>
            <td>85.1</td>
            <td>86.7</td>
            <td>74.8</td>
            <td>93.9</td>
            <td>84.2</td>
        </tr>
        <tr>
            <td>LoRA-Dash<sub>r=16</sub></td>
            <td>+ 1.3k</td>
            <td>74.8</td>
            <td>88.0</td>
            <td>80.6</td>
            <td>95.2</td>
            <td>85.6</td>
            <td>89.0</td>
            <td>77.4</td>
            <td>84.8</td>
            <td>84.4</td>
        </tr>
        <tr>
            <td>LoRA<sub>r=32</sub></td>
            <td>0.70%</td>
            <td>70.8</td>
            <td>85.2</td>
            <td>79.9</td>
            <td>91.7</td>
            <td>84.3</td>
            <td>84.2</td>
            <td>71.2</td>
            <td>79.0</td>
            <td>80.8</td>
        </tr>
        <tr>
            <td>PISSA<sub>r=32</sub></td>
            <td>+ 0</td>
            <td>67.1</td>
            <td>81.1</td>
            <td>77.2</td>
            <td>83.6</td>
            <td>78.9</td>
            <td>77.7</td>
            <td>63.2</td>
            <td>74.6</td>
            <td>75.4</td>
        </tr>
        <tr>
            <td>MiLoRA<sub>r=32</sub></td>
            <td>+ 0</td>
            <td>68.8</td>
            <td>86.7</td>
            <td>77.2</td>
            <td>92.9</td>
            <td>85.6</td>
            <td>86.8</td>
            <td>75.5</td>
            <td>81.8</td>
            <td>81.9</td>
        </tr>
        <tr>
            <td>DoRA<sub>r=32</sub></td>
            <td>+ 784k</td>
            <td>74.6</td>
            <td>89.3</td>
            <td>79.9</td>
            <td>95.5</td>
            <td>85.6</td>
            <td>90.5</td>
            <td>80.4</td>
            <td>85.8</td>
            <td>85.2</td>
        </tr>
        <tr>
            <td>LoRA-Dash<sub>r=32</sub></td>
            <td>+ 1.3k</td>
            <td>75.3</td>
            <td>88.5</td>
            <td>80.2</td>
            <td>95.7</td>
            <td>86.8</td>
            <td>90.7</td>
            <td>80.2</td>
            <td>85.6</td>
            <td>85.4</td>
        </tr>
    </tbody>
</table>

## Download for Weights

We provide some weights of our work.
| Method | Links |
| -- | -- |
| LoRA-Dash | [Google Drive](https://drive.google.com/drive/folders/1T6fVhGgO6mhjAyS8bJ7lDvWGgQGkMX6e?usp=sharing) |
| DoRA | [Github](https://github.com/NVlabs/DoRA/tree/main/commonsense_reasoning) |

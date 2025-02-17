While performing target recognition using the GLIP model, depth camera, NLP model is integrated, enabling target localization through voice input and sending robotic arm the specific position to grab targets. The GLIP model parameters need to be downloaded separately as glip_tiny_model_o365_goldg_cc_sbu.pth and placed in the glip_tinyweights folder. For BERT-base-uncased, the corresponding files must be downloaded from the paths displayed in the folder and placed accordingly. Considering the complex application environment of the robotic arm, the execution phase includes two recording sessions: The first recording captures background noise, then the second recording captures the target detection audio segment.

The project consists of two main parts:
1.Target recognition on the host machine:
The script 'glip_demo.py' is used for this purpose.
After configuring the environment, it can be executed on Ubuntu 20 or later.
2.Execution on the robotic arm:
The script arm_execution.py should be run on a Linux system machine.
Ubuntu 20.04 is recommended for optimal compatibility.


The demo.mp4 was recorded at Centre for Brain Inspired Computing Research, Department of Precision Instrument, Tsinghua University. In this demo, the speech content is: '把柠檬、橘子和苹果放到白色盆里' ('Put lemon, orange and apple in the white pot.' in English)































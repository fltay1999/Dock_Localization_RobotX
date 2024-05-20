NTU FYP: RobotX Development of Surface Vehicle's Visual Perception of Objects
- The folder named Code comes in a set, to start the program, open the Code Workspace file named Main GUI
- Open the Main GUI.py and start the program from there
- Images used for training and testing are uploaded (All images other than "Testing Images" folder is training)
- The hardware is mounted on top of the AMS but improvements are required:
  1. The cables glands are for USB 2.0, for better functionality, need USB 3.0 version
  2. The current inlet is too small, the airflow is lacking
  3. The current cooling method is an USB fan, if possible, make the enclosure waterproof with another cooling method
- The software can do localization but there are bugs:
  1. The saving of multiple classes, i.e. the possibilities repeat the same number of times as the number of classes, this         only works on manual resizing
  2. The current program only accepts one class for auto resizing
  3. The program is slow, like 1 localization takes half a minute, cannot be used for real-time
- More information is in the report

![Mounted Enclosure](https://github.com/fltay1999/Dock_Localization_RobotX/assets/142559669/1c8629d1-52d7-4905-b4e6-da479bb7f7ca)

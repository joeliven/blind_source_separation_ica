CS391L HW2 README

Joel Iventosch
jwi245
joeliven@gmail.com


INSTRUCTIONS TO EXECUTE CODE:
-unzip file into desired directory
-in command line type:   hw2ICA_Beta.py

to run program. The program will prompt you to enter your desired specifications for each parameter. Make sure to enter the data type indicated by the prompt, as user error checking is not handled. Once the final prompt is entered the program will execute. At the end of the program plots will be displayed. Note that the program will not actually terminate (and save all the necessary experimental data) until those plots are closed out.

Note: the program will create a master csv file in the current directory for tracking all experimental results. It will also create a subdirectory for each experiment that is run, in which sound files, image files (of the signal plots), and a csv file with experiment-sepcific data will all be created and saved. This file-saving functionality works very well on Windows, but has not been tested in a Linux environment.

CODE FILES:

hw2ICA_Beta.py //main program file...controls the flow of the program, etc.

hw2ICA_functions_Beta.py //helper functions called by the hw2ICA_Beta.py

sounds.mat //sound data so that you can run the program from 
//whatever directory you unzip it to.

icaTest.mat //test data
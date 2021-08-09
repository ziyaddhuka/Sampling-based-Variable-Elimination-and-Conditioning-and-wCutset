# If you are getting errors or not getting the output in PART 1 then try PART 2

# -------------PART 1-------------
> pip install -r requirements.txt
## Run the file with following as arguments 
## sampling type 'adp' for adaptive or 'smpl' for simple uniform distribution 
## sample_size
## w_cutset
## .uai file 
## .evid file

#for e.g.
> python sampling.py adp 10 1 hw3-files/Grids_14.uai hw3-files/Grids_14.uai.evid


## if you want to check the error as well then provide an extra parameter i.e. .PR file to check_sampling.py file
> python check_sampling.py adp 10 1 hw3-files/Grids_14.uai hw3-files/Grids_14.uai.evid hw3-files/Grids_14.uai.PR 




# -------------PART 2-------------
# Steps to run the code... commands are tested in linux.. you can apply alternative commands for windows/MacOS
## Step 1 creating a virtual environment to run the code so that it does not conflicts with other instaled packages on the machine
> python3 -m venv my_env
## Step 2 if the above gives error then make sure your python version is 3.6 or above and install the venv package. If no error move to Step 3
	### for linux and MacOS
	> python3 -m pip install --user virtualenv
	### for windows
	> py -m pip install --user virtualenv
	
	> python3 -m venv my_env

## Step 3 activate the environment
> source my_env/bin/activate


## Step 2 use requirements.txt file to install required packages
> pip install -r requirements.txt

## Run the file with following as arguments 
## sampling type 'adp' for adaptive or 'smpl' for simple uniform distribution 
## sample_size
## w_cutset
## .uai file 
## .evid file

## for e.g. here algo selected is adp sample size is 10 and w_cutset is 1 
> python sampling.py adp 10 1 hw3-files/Grids_14.uai hw3-files/Grids_14.uai.evid

## if you want to check the error as well then provide an extra parameter i.e. .PR file to check_sampling.py file
> python check_sampling.py adp 10 1 hw3-files/Grids_14.uai hw3-files/Grids_14.uai.evid hw3-files/Grids_14.uai.PR 



### once done with grading of the code you can deactivate the environment and delete it
> deactivate
> rm -r my_env

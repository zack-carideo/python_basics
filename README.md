# python_basics
overview of core python concepts with examples

# python_basics
overview of core python concepts with examples

### Generating new SSH key and add to the ssh-agen
  1. check if you already have ssh key present 
  ```bash
     la -al ~/.ssh
  ```
  2. if you dont have one, generate one 
  ```bash
     ssh-keygen -t ed25519 -C "zjc1002@gmail.com"
  ```
  3. add ssh key to ssh-agent (below works on windows10 dual boot 64 bit with wsl2, did this within wsl2, so watchout) 
  ```bash
     ssh-agent /bin/sh
     ssh-add ~/.ssh/id_ed25519
  ```
  4. add the ssh key to your github account 
 
 
### Creating and tracking local code in remote Gitrepo
##### Situation: code is saved on local and you need to link your local code to the repo(without loosing code)? 
  1. create a blank repo on github
  2. initalize a local repo in the directory contianing the code you want to save to the repo repo
      ```gitbash 
      git init
      ```
  4. add and commit the files you want to track to the local repo 
      ```gitbash
      git add . 
      git commit -am 'commit it all for the first time'
      ```
  5. copy the HTTPS URL of the new repo from github(here) and add the repo address as an uppstream remote repo on your local 
      ```gitbash
         git remote add origin https://github.com/zack-carideo/python_basics.git
      ```
      
  5. push the local code to the newly created remote branch 
      ```gitbash
         git push -f origin master
      ```

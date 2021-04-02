# python_basics
overview of core python concepts with examples

# Creating and tracking local code in remote Gitrepo
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

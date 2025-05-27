


0. Why Git
1. Setting up Git
	1. Download Git https://github.com/git-guides/install-git
	2. Copy Repo URL
		. ![[01_Copy_Link.png]]
	3. Clone Repo in Terminal : git clone https://github.com/mgalkowski/fluxy-mpi-bgc.git
		![[02_git_clone.png]]
	4. Caching GitHub credentials in Git
		1. [Install](https://github.com/cli/cli#installation) GitHub CLI (you can also use this command in the terminal: `sudo apt install gh`)
		2. `gh auth login` (You will be asked a few questions, select the following)
			1. GitHub.com
			2. HTTPS
			3. Yes
			4. Login with web browser ( You then have to copy the code from the terminal and enter it in the new internet-tab)
2. Git Workflow
	1. Starting point: https://github.com/mgalkowski/fluxy-mpi-bgc
	2. Set up an Issue(if it not already exists) and describe what should be done here.
		Issues -> New issue
	3. Assign yourself or someone else to an Issue
		Choose an issue -> Assignees -> Assign yourself
	4. Create a branch 
		and make sure the Repository destination is mgalkowski/fluxy-mpi-bgc and the Branch source is main-mpi-bgc ( see picture)
		Development -> Create branch
		![[03_Create_branch.png]]
	5. Checkout you local repository
		1. Open your terminal or console
		2. Change to the Repository
			`cd path/to/repository 
		3. `git fetch origin
		4. `git checkout your-new-branch
	6. Start working on your issue
	7. Commit often 
		(small, focused changes) & write clear and short commit messages
		1. Add your changes that you want to commit
			1. `git add .` (Adds all changed files) or
			2. `git add path/to/file` (Adds only the selected file)
		2. `git commit -m"Clear and short commit message" `  
	8. Push commits at the end of the day 
		to the upstream repository ( https://github.com/mgalkowski/fluxy-mpi-bgc.git)
		`git push`  
 3. How to do a Pull Request 
		





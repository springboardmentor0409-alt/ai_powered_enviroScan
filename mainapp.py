STEP 1: GitHub rejected your push (file > 100 MB)

GitHub does not allow files larger than 100 MB, so it blocked your push.

🔹 STEP 2: Install Git LFS
git lfs install

🔹 STEP 3: Tell Git LFS to track large file types

Example:

git lfs track "*.csv"
git lfs track "*.joblib"
git lfs track "*.png"


Git creates a .gitattributes file.

🔹 STEP 4: Add changes and commit
git add .
git commit -m "Track large files using Git LFS"

🔹 STEP 5: GitHub still rejects the push

Reason:
❗ Old commits still contain the large files.
So even after LFS, GitHub blocks the push.

🔹 STEP 6: Rewrite history using Git LFS migrate

This removes large files from old commits:

git lfs migrate import --include="*.csv,*.joblib,*.png"

🔹 STEP 7: Force push rewritten history
git push origin branch-name --force

🔹 STEP 8: Done — now normal workflow

From now on, use:

git add .
git commit -m "message"
git push origin branch-name


Files will automatically go into Git LFS next time.







git rm --cached dashboard/random_forest/random_forest.joblib
git rm --cached data/labeled_pollution_data.csv



These are just a few steps I tried for fixing the GitHub large file error. I’m not fully sure if everything is correct, but these are the commands I used.



# 1) Commit the staged .gitignore
git commit -m "Ignore large model & data files"

# 2) Create a new branch from current HEAD
git checkout -b new_branch

# 3) Verify the large files are not present in the working tree (quick check)
git ls-files | Select-String "models/random_forest|random_forest_model.pkl|labeled_pollution_data.csv" -NotMatch

# 4) Push the new branch to origin (no force)
git push origin new_branch
try yhis
    



(venv) D:\collab_folder\ai_powered_enviroScan>git push origin thanvi_project
Enumerating objects: 13, done.
Counting objects: 100% (13/13), done.
Delta compression using up to 8 threads
Compressing objects: 100% (6/6), done.
Writing objects: 100% (7/7), 3.61 KiB | 1.20 MiB/s, done.
Total 7 (delta 4), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (4/4), completed with 4 local objects.
remote: error: GH013: Repository rule violations found for refs/heads/thanvi_project.
remote: Review all repository rules at https://github.com/springboardmentor0409-alt/ai_powered_enviroScan/rules?ref=refs%2Fheads%2Fthanvi_project
remote:
remote: - Changes must be made through a pull request.
remote:
remote: - Cannot update this protected ref.
remote:
To https://github.com/springboardmentor0409-alt/ai_powered_enviroScan.git
 ! [remote rejected]   thanvi_project -> thanvi_project (push declined due to repository rule violations)
error: failed to push some refs to 'https://github.com/springboardmentor0409-alt/ai_powered_enviroScan.git'

(venv) D:\collab_folder\ai_powered_enviroScan>git branch
  main
* thanvi_project

(venv) D:\collab_folder\ai_powered_enviroScan>git push origin thanvi_project
Enumerating objects: 13, done.
Counting objects: 100% (13/13), done.
Delta compression using up to 8 threads
Compressing objects: 100% (6/6), done.
Writing objects: 100% (7/7), 3.61 KiB | 1.80 MiB/s, done.
Total 7 (delta 4), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (4/4), completed with 4 local objects.
remote: error: GH013: Repository rule violations found for refs/heads/thanvi_project.
remote: Review all repository rules at https://github.com/springboardmentor0409-alt/ai_powered_enviroScan/rules?ref=refs%2Fheads%2Fthanvi_project
remote:
remote: - Changes must be made through a pull request.
remote:
remote: - Cannot update this protected ref.
remote:
To https://github.com/springboardmentor0409-alt/ai_powered_enviroScan.git
 ! [remote rejected]   thanvi_project -> thanvi_project (push declined due to repository rule violations)
error: failed to push some refs to 'https://github.com/springboardmentor0409-alt/ai_powered_enviroScan.git'
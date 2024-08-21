# Today I Learned ...

## Setting up GitHub to work with Enterprise and private repos is a bit of a pain
I think that all of the hassle I had would have been avoided if I had just used the GitHub CLI on the MacBook from the start.

`gh auth login` prompts whether you want to authenticate with GitHub.com or Enterprise.

My issue was that I cloned my repo whilst still authed in the Enterprise GitHub, so when I tried to push my changes I got an error about not being able to access the repo.

I had to configure my `.ssh/config` file to use the correct ssh key for my PA and Enterprise GitHub accounts.

```bash
# Work GitHub account
Host github.com-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa_work  # Replace with your actual work SSH key path
    IdentitiesOnly yes

# Personal GitHub account
Host github.com-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa_personal  # Replace with your actual personal SSH key path
    IdentitiesOnly yes
```

Then, I had to setup a fine-grained personal access token for my PA on the repo I was working on.

Then, `gh auth login` to get logged in to the PA.

Only then did it finally work.  (I went through a bunch of extra hassle to distill this process down to this.)
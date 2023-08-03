#!/bin/bash

mkdir -p /root/.ssh/
mv setup/id_rsa_github_droidaugmentordeploy /root/.ssh/
mv setup/id_rsa_github_droidaugmentordeploy.pub /root/.ssh/
mv setup/config /root/.ssh/
chown -R root /root/.ssh
ssh -o StrictHostKeyChecking=no -T git@github.com
git clone -b docker git@github.com:LEA-SF23/DroidAugmentor.git
rm -f $0

---
title: Github上fork项目后与源项目同步
---
查看当前的远程仓库地址：
```bash
git remote -v
```

添加一个别名为upstream（上游）的地址，指向fork的原仓库地址
```bash
git remote add upstream https://github.com/XXX/XXXX
```

保持本地仓库和上游仓库同步：
```bash
git fetch upstream
git checkout master
git merge upstream/master
```

推送本地仓库到远程仓库
```bash
git push origin master
```

参考：[保持fork之后的项目和上游同步](https://github.com/staticblog/wiki/wiki/%E4%BF%9D%E6%8C%81fork%E4%B9%8B%E5%90%8E%E7%9A%84%E9%A1%B9%E7%9B%AE%E5%92%8C%E4%B8%8A%E6%B8%B8%E5%90%8C%E6%AD%A5)

README
---
Code & data to reproduce the analyses in our ACL 2014 paper: 

    Humans Require Context to Infer Ironic Intent (so Computers Probably do, too)
        Byron C Wallace, Do Kook Choe, Laura Kertz, and Eugene Charniak

Made possible by support from the Army Research Office (ARO), grant# 528674 
"Sociolinguistically Informed Natural Language Processing: Automating Irony Detection"

Contact: Byron Wallace (byron.wallace@gmail.com)

* The actual database - a flat sqlite file - is ironate.db.zip, it needs to be unzipped, of course. 
* The database-schema.txt file (in this directory) contains information regarding the database.
* See irony_stats.py for instructions on how to reproduce our analyses. This also gives examples on working with the database in python (and in SQL, since we issue queries directly). Note that this requires sklearn, numpy, & statsmodels modules to be installed.






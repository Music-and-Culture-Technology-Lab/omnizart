Add New Modules
===============

This page describes how to add new modules into Omnizart project, adapt the original implementations
to omnizart's architecture.

Before starting walking through the integration process, be sure you have already read the
`CONTRIBUTING.md <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/CONTRIBUTING.md>`_,
and the `slides of omnziart <https://drive.google.com/file/d/1IO1lh07nMvSi0X0nzRDT7kuE1f468Rl1/view?usp=sharing>`_
could also be helpful for your understanding of this project.
Additionally, there are few more things to be always kept in mind while developing omnizart.

Principles
##########

* **Find an existing module and start development** - There are already several implemented modules
  that are fully functional, and being great examples that give you hints on your way developing
  new modules. Most of them are very similar of their overall architecture, but vary in detail.
  Most the time, you could just copy and paste the small pieces to your module, and modify just a
  small part of them to adapt to your task.
* **Try not to make your own wheels** - There have been many useful and validated functions that are
  developed to deal with the daily works. They are already there to cover 90% of every details of a
  module, thus new logics are in very small chances being needed. 
  Most of the time you need to implement the most would be the part of feature and label extraction,
  which will be explained in the upcoming sections.
* **Check with linters frequently** - You should always do ``make lint`` before you push to github,
  checking that there aren't any errors with the code format, or the build process would fail.
* **Don't permit linter errors easily** - You may find some comments that permits the linter errors
  while surfing the code. Those are quick solutions while in the early development of omnizart, which
  saves lots of time fixing those lint errors. But it should not be the main concern now, as the
  architecture is more stable and less error prone. You should follow every hints by the linters
  and fix them before you file a pull request.
 

----

So now we are all set and ready to add a new module to omnizart. Here we will take the 
`PR #11 <https://github.com/Music-and-Culture-Technology-Lab/omnizart/pull/11>`_ as the example.

Setup
#####

Implement Feature Generation
############################

Implement Model Training
########################

Implement Transcription
#######################

Add Unit Tests
##############

Commit Checkpoints
##################

Implement CLI
#############

Add Documentation
#################

----

Optional
########

This section holds the optional actions you can do, while it is not necessary to be done
during implementing a new module.

Add new supported datasets
**************************

If you want to add a new dataset that is currently not supported by ``omnizart`` (which is defined in
:class:`omnizart.constants.datasets`), things should be noticed are explained in this section.

(To be continue...)


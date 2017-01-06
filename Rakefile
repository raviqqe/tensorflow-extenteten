require_relative './third/tensorflow-rakefile/tfrake.rb'
include TFRake


define_tasks('extenteten',
             pytest_flags: ['--ignore', 'extenteten/_experimental'])


task :test => :pytest

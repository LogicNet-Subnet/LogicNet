pm2 install pm2-logrotate

pm2 set pm2-logrotate:max_size 10M
pm2 set pm2-logrotate:compress false
# Rotate every 10 minutes
pm2 set pm2-logrotate:rotateInterval '*/10 * * * *'  
pm2 set pm2-logrotate:retain 100
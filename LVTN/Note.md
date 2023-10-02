Enable Docker
1. `dism.exe /Online /Enable-Feature:Microsoft-Hyper-V /All`
2. Enable Windows Hypervisor Platform
3. ```bcdedit /set hypervisorlaunchtype auto```

Disable Docker for game
1. `dism.exe /Online /Disable-Feature:Microsoft-Hyper-V
2. Disable Windows Hypervisor Platform
3. ```bcdedit /set hypervisorlaunchtype off```

Connect postgres
```docker exec -it chatbox-postgres psql -U postgres```

create database mydb;  
create user myuser with encrypted password 'mypass';  
grant all privileges on database mydb to myuser;

Start rasa
- `rasa run actions`
- ``rasa interactive`` or ``rasa run --model models --enable-api --cors “*”``

https://azrotv.com/extras/sms-verification/messages.php?id=4489

Taichi: Sáng - nắng (rừng đen)
Cá voi: Sáng - nắng (núi tuyết)
Kappa: Trời mưa là được (gia viên, rừng mộng)
Rái cá: Sáng - mưa (núi tuyết)
Cá mập: Đêm - mưa (rừng mộng)
Killi ma: Đêm - mưa (bãi cát)

0935792847
from locust import HttpUser, between, task

class ApplicationUser(HttpUser):
    wait_time = between(5, 15)

    @task
    def load_main(self):
        self.client.get("")
    @task
    def create_project(self):
        self.client.post("/userproject",data={'name':'test'})
    @task
    def upload_file(self):
        self.client.post("/upload",files={'file':open('train1.csv','rb')})
    @task
    def load_file(self):
        self.client.get("/filedata/204")


# temporal-causal-pi-detection


Qwen-3-max没有Embeddings,这时候需要切换到pumpkin（或 OpenAI 官方）
<img width="1432" height="974" alt="image" src="https://github.com/user-attachments/assets/42591d7d-52df-4815-972a-608a175aac15" />


然后我们的TCDT方法跑的时候，有时候还会遇到并发key的问题，就是导致流程结束。因为重执行了多次，如下：

<img width="2454" height="1260" alt="image" src="https://github.com/user-attachments/assets/0c483836-9b77-4e0f-891b-e53d36ef81bd" />

<img width="1974" height="936" alt="d8d74b838a954b4bd1bfaf61b6e872e1" src="https://github.com/user-attachments/assets/7b07011a-f292-452b-a473-074a3c2bbc1a" />


因此需要改一下llm的接口，让他不崩溃



<img width="1728" height="710" alt="image" src="https://github.com/user-attachments/assets/850bd5b9-ff30-45d7-a887-4fc7e41479bc" />


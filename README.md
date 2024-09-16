![header.png](media/header.png)
# 04 Continous User Feedback
## Streamlit App
Install pip environment via `pip install -r requirements.txt`

Run with 
`streamlit run badpun.py` from root directory.
When running two instances, specify port by `streamlit run badpun.py --server.port 8051`. For one high clustering and 
one low clustering instance run
```
$ streamlit run badpun.py --server.port 8051 -- high
$ streamlit run badpun.py --server.port 8052 -- low
```

### Structure
The folder `src/` contains all files code files required for the Application. `experiments/` contains jupyter notebooks that
were used to determine parameters and conduct experiments. In the folder `data/` lies both example data for training of the model and for running the application.
The folder `submission/` contains our final report pdf file.

### Environment
The file 'config.ini' contains all constants used throughout the system. In order to avoid spread of information, best
add new constants to this file, as well as parameterization of methods etc. Usage:

``` 
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
config['DATA']['TestUserEmbeddingPath']
```

### Branches
On the `main` branch, the wordclouds are computed using the simple attentions due to time constraints.  For the LRP implementation please switch to the `high-dim-and-lrp` branch.
`
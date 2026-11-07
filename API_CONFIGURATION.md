# API configuration

The API allows browser clients from the origins listed in
`DOBBY_CORS_ORIGINS`. Set it to a comma-separated list without spaces in the
origin values, for example:

```bash
DOBBY_CORS_ORIGINS=https://dashboard.example.com,https://admin.example.com
```

When the variable is unset, local development permits
`http://localhost:3000` and `http://localhost:5173`. Wildcard origins are not
accepted because the API enables credentialed CORS requests.

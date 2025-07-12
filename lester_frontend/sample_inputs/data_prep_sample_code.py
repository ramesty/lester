dataprep_original_code = """ Not Using this for now          
"""

dataprep_input_arg_names = ['customers_file', 'mails_file']

dataprep_input_schemas = [['customer_id', 'customer_email', 'bank', 'country', 'level'],
                ['mail_id', 'email', 'raw_date', 'mail_subject', 'mail_text']]

dataprep_output_columns = ['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']
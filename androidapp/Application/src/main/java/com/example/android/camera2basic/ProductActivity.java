package com.example.android.camera2basic;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import static com.example.android.camera2basic.R.id.tableLayout;


public class ProductActivity extends Activity
{

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_product);
        getWindow().getDecorView().setBackgroundColor(Color.WHITE);

        // Get the Intent that started this activity and extract the string
        Intent intent = getIntent();
        String fileName = intent.getStringExtra("IMAGE");

        Bitmap btmp = BitmapFactory.decodeFile(fileName, null);
        //btmp = Bitmap.createScaledBitmap(btmp, 384, 216, false);

        TableLayout tl = (TableLayout) findViewById(tableLayout);
        //tl.getRootView().setBackgroundColor(getResources().getColor(android.R.color.white));

        String URL = "https://obormot-service.cfapps.io";
        Connection conn = new Connection();
        conn.setBasicUrl(URL);
        try
        {
            String ret = conn.uploadFile(btmp, "somefile.jpg");
            String[] list = ret.split(".jpg");
            for(String imgfile : list)
            {
                String image_url = URL + "/" + imgfile + ".jpg";
                btmp = conn.getBitmapFromURL(image_url);
                String descr = imgfile.replace("images/", "").replace(".jpg", "");
                int pos = descr.indexOf('/');
                descr = descr.substring(pos + 1, descr.length());
                if (null != btmp)
                {
                    btmp = Bitmap.createScaledBitmap(btmp, 600, 780, false);
                    TableRow tr = new TableRow(this);
                    ImageView iv = new ImageView(this);
                    iv.setImageBitmap(btmp);
                    tr.addView(iv);
                    tl.addView(tr);
                    // add description
                    tr = new TableRow(this);
                    tl.addView(tr);

                    LinearLayout ll = new LinearLayout(this);
                    ll.setOrientation(LinearLayout.VERTICAL);
                    ll.setGravity(Gravity.CENTER);
                    tr.addView(ll);

                    TextView tv = new TextView(this);
                    tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
                    tv.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT));
                    tv.setSingleLine(false);
                    tv.setText(descr);
                    ll.addView(tv);

                    Button btn = new Button(this);
                    btn.setText("Buy!");
                    LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT);
                    btn.setLayoutParams(lp);
                    ll.addView(btn);
                }
            }
        }
        catch(Exception e)
        {
            Log.d("DEBUG", "error reading service");
        }

    }
}

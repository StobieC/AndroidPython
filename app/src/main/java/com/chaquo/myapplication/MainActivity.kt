package com.chaquo.myapplication

import android.content.Context
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.PyException
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        val py = Python.getInstance()
        val module = py.getModule("plot")

        findViewById<Button>(R.id.button).setOnClickListener {
            try {
                val bytes = module.callAttr("plot",
                                            findViewById<EditText>(R.id.etX).text.toString(),
                                            findViewById<EditText>(R.id.etY).text.toString())
                    .toJava(ByteArray::class.java)
                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                findViewById<ImageView>(R.id.imageView).setImageBitmap(bitmap)


                val imageAsString = stringImage(bytes)

                val rectModule = py.getModule("rectification")


                val test = rectModule.callAttr("test")
                findViewById<TextView>(R.id.test).text = test.toString()
                Log.d("PyObj", "$test")

                rectModule.callAttr("rectify_image", imageAsString).toJava(ByteArray::class.java)

                currentFocus?.let {
                    (getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager)
                        .hideSoftInputFromWindow(it.windowToken, 0)
                }
            } catch (e: PyException) {
                Toast.makeText(this, e.message, Toast.LENGTH_LONG).show()
            }
        }
    }


    fun stringImage(bytes: ByteArray): String {
        return Base64.encodeToString(bytes, Base64.DEFAULT)
    }
}
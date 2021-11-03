import React from "react"
import Layout from "../components/layout"
import SEO from "../components/seo"
import style from "./products.module.css"
import { Link } from "gatsby"
const ProductsPage = () => {
  return (
    <Layout>
      
      <SEO
        title="Products"
        description="JumpML Products"
        image="/jumpML.svg"
        pathname="/products"
        // Boolean indicating whether this is an article:
        // article
      />
      
      <section className={style.wrapper}>
      <h2 className={style.heading}>JumpML Noise Reduction</h2>
        <h1> Problem and motivation </h1>
        

        <li className={style.flexcontainer}>
        <ul className={style.ul}>
        Noisy World
        <img
            src="noises/leaf_blower.svg"
            width="300"
            height="274"
            alt="LeafBlower"
            className={style.center}
          />
          </ul>
          <ul className={style.ul}>
           My Life
          <img
            src="video_chat.svg"
            width="366"
            height="374"
            alt="VideoChat"
            className={style.center}
          />
          </ul>
          <ul className={style.ul}>
           Going on mute
          <img
            src="distressed_man.svg"
            width="366"
            height="374"
            alt="Headache"
            className={style.center}
          />
          </ul>
          <li className={style.flexitem}> <h1> The only constant in life is change... and background noise. </h1> </li>
          </li>

          <h1> Our Solution </h1>

          <li className={style.flexcontainerAnnounce}>
          <ul className={style.ul}>
          JumpML Noise Reduction powered by SignalSifter

          <img
            src="JumpML_NoiseReduction.png"
            alt="JumpML_NR"
          />

          <li className={style.flexitem}> <h1> Cut the background noise 
        and preserve the speech, so you are heard clearly </h1> </li>
          </ul>
          </li>

          <h1> Product benefits </h1>

          <li className={style.flexcontainer}>
          
          <ul className={style.ul}>
            Strong Performance
          <img
            src="strong_brain.svg"
            alt="Performance"
            className={style.center}
          />
          <li className={style.flexitem}>Cuts diverse noises</li>
          <li className={style.flexitem}>Improve intelligibility</li>
          <li className={style.flexitem}>Improve MOS/WER</li>
          <li className={style.flexitem}>SignalSifter optimized for audio</li>
          </ul>
          <ul className={style.ul}>
            Energy-efficient
          <img
            src="energy-efficient.svg"
            alt="energy-efficient"
            className={style.center}
          />
          <li className={style.flexitem}>Made for embedded</li>
          <li className={style.flexitem}>Operates at 16kHz</li>
          <li className={style.flexitem}>Uses efficient RNN</li>
          </ul>
          <ul className={style.ul}>
            Low latency
          <img
            src="lightning.svg"
            alt="LowLatency"
            className={style.center}
          />
          <li className={style.flexitem}>10 ms or less</li>
          <li className={style.flexitem}>Useful for hearing aids</li>
          <li className={style.flexitem}>Smart transparency</li>   
          </ul>
          <li className={style.flexitem}> <h1> Forget about the noise and speak with ease, anywhere and anytime.  </h1> </li>
          {/* <ul className={style.ul}>
            Easy Deployment
            <img
            src="gears.svg"
            alt="Deployment"
            className={style.center}
          />
          <li className={style.flexitem}>Simple C code</li>
          <li className={style.flexitem}>Easy to use API</li>
          <li className={style.flexitem}>No external libraries/frameworks!</li>
          </ul>

          <ul className={style.ul}>
            Always Improving
            <img
            src="business-growth.svg"
            alt="Memory"
            className={style.center}
          />
          <li className={style.flexitem}>Speech/Noise database growing </li>
          <li className={style.flexitem}>Process for issue resolution</li>
          <li className={style.flexitem}>Innovate and keep track of research</li>
          </ul> */}
        </li>


        <h1> Comparison Table </h1>


  <table className={style.table}>
  <tr >
    <th>Features</th>
    <th>JumpML NR</th>
    <th>RNNoise</th>
    <th>Other offerings</th>
  </tr>
  <tr className={style.table_rowA}>
    <td>Good noise reduction </td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="plum">Depends</td>
    <td bgcolor="palegreen">Yes</td>
  </tr>
  <tr className={style.table_rowB}>
    <td>Model/Memory &lt; 0.4 MB </td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="pink">No</td>
  </tr>
  <tr className={style.table_rowA}>
    <td>Low-latency &lt;= 10 ms </td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="pink">No</td>
  </tr>
  <tr className={style.table_rowB}>
    <td>Real-time on tiny embedded </td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="pink">No</td>
  </tr>
  <tr className={style.table_rowA}>
    <td>Privacy (no internet needed) </td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="pink">No</td>
  </tr>
  <tr className={style.table_rowB}>
    <td>Portable/inspectable C code</td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="pink">No</td>
  </tr>
  <tr className={style.table_rowA}>
    <td>Adjustable Noise Reduction</td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="plum">Easy</td>
    <td bgcolor="pink">No</td>
  </tr>
  <tr className={style.table_rowB}>
    <td>Customizable</td>
    <td bgcolor="palegreen">Yes</td>
    <td bgcolor="pink">No</td>
    <td bgcolor="pink">No</td>
  </tr>
</table>

<h1> More Info </h1>
<p>
For JumpML Noise Reduction audio demos and specific use cases, please checkout our <Link to="/solutions">
        Use Cases</Link>{" "} page.
</p>

<p>
For questions about how it can fit into your product, or about what customizations are available for your
use case, or if you just want more details about the implementation, or just to say hello, please contact us at{" "}
          <a href="mailto:info@jumpml.com">JumpML email.</a>
</p>

      </section>
    </Layout>
  )
}

export default ProductsPage

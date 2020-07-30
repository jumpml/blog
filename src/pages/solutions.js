import React from "react"
import Layout from "../components/layout"
import SEO from "../components/seo"
import style from "./solutions.module.css"

const SolutionsPage = () => {
  return (
    <Layout>
      <SEO
        title="Our Solutions"
        description="ML+DSP Algorithm Solutions"
        image="/jumpML.svg"
        pathname="/solutions"
        // Boolean indicating whether this is an article:
        // article
      />
      <section className={style.wrapper}>
        <h2>Our Solutions</h2>

        <h3> Energy-efficient, scalable, low-latency</h3>

        <li className={style.flexcontainer}>
          <img
            src="jumpML-audio.svg"
            width="366"
            height="374"
            alt="JumpML voice"
            className={style.site_logo}
          />
          <ul className={style.ul}>
            soundClean
            <li className={style.flexitem}>Echo cancelation</li>
            <li className={style.flexitem}>BF + Dereverb</li>
            <li className={style.flexitem}>Noise suppression</li>
          </ul>
          <ul className={style.ul}>
            soundCatch
            <li className={style.flexitem}>Keyword Spotting + SV </li>
            <li className={style.flexitem}>Speech Recognition</li>
            <li className={style.flexitem}>Sound Event Localization</li>
          </ul>
          <ul className={style.ul}>
            soundPlay
            <li className={style.flexitem}>Text-to-Speech</li>
            <li className={style.flexitem}>MBDRC and EQ</li>
            <li className={style.flexitem}>Smart Volume</li>
          </ul>
        </li>

        {/* <li className={style.listitem}>
          <img
            src="jumpML-vision.svg"
            width="366"
            height="374"
            alt="JumpML vision"
            className={style.site_logo}
          />
          <li> Test</li>
        </li> */}
      </section>
    </Layout>
  )
}

export default SolutionsPage
